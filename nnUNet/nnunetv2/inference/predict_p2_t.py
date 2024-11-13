import inspect
import itertools
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
import copy
base = '/media/bit301/data/yml/project/python39/p2/nnUNet/DATASET_p2'
nnUNet_raw = join(base, 'nnUNet_raw') # os.environ.get('nnUNet_raw')
# nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # os.environ.get('nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results_p2x2') # os.environ.get('nnUNet_results')

# from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import SimpleITK as sitk
import time
from skimage.measure import regionprops
import h5py
from skimage import measure, morphology
from skimage.morphology import binary_closing
import cv2
from scipy.ndimage import label

class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            # device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!
            # why would I ever want to do that. Stupid dobby. This kills DDP inference...
            pass
        if device.type != 'cuda':
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                           num_input_channels)
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)#注意修改mask，因为实际训练为2
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)
        # preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        # # hijack batchgenerators, yo
        # # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # # way we don't have to reinvent the wheel here.
        # num_processes = max(1, min(num_processes, len(input_list_of_lists)))
        # ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
        #                         output_filenames_truncated, self.plans_manager, self.dataset_json,
        #                         self.configuration_manager, num_processes)
        # if num_processes == 0:
        #     mta = SingleThreadedAugmenter(ppa, None)
        # else:
        #     mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
        # return mta

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    # print('sleeping')
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
                    #                               dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(prediction, plans_manager,
                    #                                                             configuration_manager, label_manager,
                    #                                                             properties,
                    #                                                             save_probabilities)
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        with torch.no_grad():
            prediction = None

            for params in self.list_of_parameters:

                # messing with state dict names...
                if not isinstance(self.network, OptimizedModule):
                    self.network.load_state_dict(params)
                else:
                    self.network._orig_mod.load_state_dict(params)

                # why not leave prediction on device if perform_everything_on_device? Because this may cause the
                # second iteration to crash due to OOM. Grabbing tha twith try except cause way more bloated code than
                # this actually saves computation time
                if prediction is None:
                    prediction = self.predict_sliding_window_return_logits(data).to('cpu')
                else:
                    prediction += self.predict_sliding_window_return_logits(data).to('cpu')

            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)

            if self.verbose: print('Prediction done')
            prediction = prediction.to('cpu')
        torch.set_num_threads(n_threads)
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        # prediction = self.network(x)
        prediction = self.network(x)[0]#######

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)
            ]
            for axes in axes_combinations:
                # prediction += torch.flip(self.network(torch.flip(x, (*axes,))), (*axes,))
                prediction += torch.flip(self.network(torch.flip(x, (*axes,)))[0], (*axes,)) #####
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        results_device = self.device if do_on_device else torch.device('cpu')

        # move data to device
        if self.verbose: print(f'move image to device {results_device}')
        data = data.to(self.device)

        # preallocate arrays
        if self.verbose: print(f'preallocating results arrays on device {results_device}')
        predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                       dtype=torch.half,
                                       device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
        if self.use_gaussian:
            gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                        value_scaling_factor=10,
                                        device=results_device)
        empty_cache(self.device)

        if self.verbose: print('running prediction')
        if not self.allow_tqdm and self.verbose: print(f'{len(slicers)} steps')
        for sl in tqdm(slicers, disable=not self.allow_tqdm):
            workon = data[sl][None]
            workon = workon.to(self.device, non_blocking=False)

            prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
            # print(predicted_logits[sl].shape)
            predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
            n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)

        predicted_logits /= n_predictions
        # check for infs
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                               'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                               'predicted_logits to fp32')
        return predicted_logits

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose: print(f'Input shape: {input_image.shape}')
                if self.verbose: print("step_size:", self.tile_step_size)
                if self.verbose: print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
                    except RuntimeError:
                        print('Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
        return predicted_logits

def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')


    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=1, part_id=0)

def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=False,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=args.num_parts,
                                 part_id=args.part_id)
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)

def postpossess(img_array,mask_array):
    mask = copy.deepcopy(mask_array)
    # process
    mask[mask > 0] = 1
    img_array1 = img_array * mask  # 勾画区域
    # img_array1[img_array1>130]=1#钙化
    img_array1 = np.where(img_array1 > 130, 1, 0)
    inverted_mask = 1 - img_array1
    mask_array[mask_array == 2] = 1  # 让钙化也先划分为管腔
    mask_array = mask_array * inverted_mask  # 腾出钙化区域
    mask_array = mask_array + img_array1 * 2  # 钙化为2
    return mask_array

def get_longest_3d_mask(mask):
    mask1=copy.deepcopy(mask)
    mask=np.where(mask>0,1,0)
    # Step 1: Extract connected components
    labeled_mask, num_features = label(mask)

    if num_features == 0:
        return None  # No connected components found

    # Step 2: Compute geometric features and select the longest component
    max_length = 0
    longest_component = None
    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i).astype(np.uint8)

        # Calculate the length using the main axis of the bounding box
        props = regionprops(component_mask)
        if len(props) > 0:
            bbox = props[0].bbox
            # Assuming the bounding box is sorted in ascending order
            length = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])

            # Alternatively, you can calculate the distance between the centers of mass
            # to estimate the length as the longest diagonal
            # center_of_masses = [center_of_mass(component_mask)]
            # length = max([euclidean(cm1, cm2) for cm1, cm2 in zip(center_of_masses[:-1], center_of_masses[1:])])

            if length > max_length:
                max_length = length
                longest_component = component_mask

    return mask1*longest_component

def remove_regions(mask):
    mask1=copy.deepcopy(mask)
    mask1=np.where(mask1>0,1,0)
    # mask2 = mask
    # mask2=np.where(mask2>1,1,0)
    segmentation_sitk = sitk.GetImageFromArray(mask1)
    # 计算标签图像的连通分量
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_components_filter.Execute(segmentation_sitk)

    # 获取每个连通分量的大小
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labeled_image)

    # 找到最大的连通分量ID
    max_size = 0
    largest_label = 0
    for i in range(1, label_shape_filter.GetNumberOfLabels() + 1):  # Label index starts from 1
        if label_shape_filter.GetNumberOfPixels(i) > max_size:
            max_size = label_shape_filter.GetNumberOfPixels(i)
            largest_label = i

    # 仅保留最大连通分量
    binary_mask = sitk.Equal(labeled_image, largest_label)
    cleaned_segmentation = sitk.Cast(binary_mask, segmentation_sitk.GetPixelID())
    cleaned_segmentation = sitk.GetArrayFromImage(cleaned_segmentation)
    cleaned_segmentation=cleaned_segmentation*mask
    # print(cleaned_segmentation.max())
    return cleaned_segmentation.astype(np.int16)

def remove_small_volums(mask):
    mask1 = np.where(mask > 0, 1, 0)
    segmentation_sitk = sitk.GetImageFromArray(mask1)
    # 计算标签图像的连通分量
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_components_filter.Execute(segmentation_sitk)
    # 获取每个连通分量的大小
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labeled_image)
    # 初始化一个空的数组来存储处理后的mask
    cleaned_segmentation = np.zeros_like(mask1)
    # 遍历每个连通分量,保留体积大于min_volume的连通区域
    for i in range(1, label_shape_filter.GetNumberOfLabels() + 1):
        if label_shape_filter.GetNumberOfPixels(i) >= 14000:
            binary_mask = sitk.Equal(labeled_image, i)
            binary_mask_array = sitk.GetArrayFromImage(binary_mask)
            cleaned_segmentation[binary_mask_array == 1] = 1
    # 返回处理后的mask
    cleaned_segmentation = cleaned_segmentation * mask
    return cleaned_segmentation.astype(np.int16)

def caw(mask):
    """
    使用矢量化操作重新赋值3D掩模中的像素值。

    :param mask: 一个三维numpy数组,代表3D掩模。
    :return: 重新赋值后的3D掩模。
    """
    reassigned_mask = np.zeros_like(mask)

    # 数值区间及其对应的赋值
    reassigned_mask[(mask >= 130) & (mask <= 199)] = 1
    reassigned_mask[(mask >= 200) & (mask <= 299)] = 2
    reassigned_mask[(mask >= 300) & (mask <= 399)] = 3
    reassigned_mask[mask >= 400] = 4

    return reassigned_mask

def diameter_area(mask,ncct_arrayy):
    # lumen_mask = np.where(mask == 1, 1, 0)  # 管腔（正常血液流动区域）
    v1=copy.deepcopy(mask)
    z_dims = mask.shape[0]
    area_per_lumen = []
    diameter_per_lumen = []

    for slice_idx in range(z_dims):  # 如果存在升主动脉,去掉升主动脉
        slice_idx=z_dims-slice_idx-1
        slice_lume = mask[slice_idx, :, :]
        img_label, num = measure.label(slice_lume, connectivity=2, return_num=True)#填充空洞
        props = measure.regionprops(img_label)
        props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
        if len(props_sorted) > 1 and props_sorted[0].area > 300 and slice_idx > 0:
            if props_sorted[1].area>300:
                v1[slice_idx+1:, :, :]=0
                v1=remove_regions(v1)
                v1[slice_idx+1:, :, :]=mask[slice_idx+1:, :, :]
                mask=copy.deepcopy(v1)
                continue#处理完就结束,节约时间
    ncct_arrayyy=ncct_arrayy*mask
    sum_z_axis = np.sum(ncct_arrayyy, axis=(1, 2))

    flag_area=np.ones([512,512])
    for slice_idx in range(z_dims):  # 获取当前切片
        slice_lume = mask[slice_idx, :, :]
        if slice_lume.sum()==0:
            diameter_per_lumen.append(0)
            area_per_lumen.append(0)  # 将面积添加到列表中
        else:
            img_label, num = measure.label(slice_lume, connectivity=2, return_num=True)
            props = measure.regionprops(img_label)
            props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
            max_label = props_sorted[0].label

            if len(props_sorted) > 1 and props_sorted[1].area>300:
                if props_sorted[0].area<2*props_sorted[1].area:
                    max_label0 = props_sorted[0].label
                    slice_lume0 = (img_label == max_label0).astype(int)
                    overlap0 = np.logical_and(slice_lume0, flag_area)
                    overlap_area0 = np.count_nonzero(overlap0)

                    max_label = props_sorted[1].label
                    slice_lume = (img_label == max_label).astype(int)
                    overlap = np.logical_and(slice_lume, flag_area)
                    overlap_area = np.count_nonzero(overlap)
                    if overlap_area>overlap_area0:
                        max_label = props_sorted[1].label
                    else:
                        max_label = props_sorted[0].label

            slice_lume = (img_label == max_label).astype(int)
            flag_area=slice_lume
            filled_slice_lume = binary_closing(slice_lume)  #空洞填充,以便于计算直径
            gray_img = np.uint8(filled_slice_lume * 255)
            contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dist_map = cv2.distanceTransform(gray_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            _, radius, _, center = cv2.minMaxLoc(dist_map)
            diameter = 2 * radius
            # threshold = np.percentile(distances, 2)
            area = np.sum(slice_lume > 0)  # 计算填充前的面积
            if area > np.pi * (diameter / 2) ** 2:
                area = np.pi * diameter ** 2
            diameter_per_lumen.append(diameter)
            area_per_lumen.append(area)

    return diameter_per_lumen, area_per_lumen,sum_z_axis

def caculate_index(mask_image,ncct_array,path_save):
    mask_image[mask_image>2]=3
    Ca = copy.deepcopy(mask_image)  # 钙化
    reassigned_Ca = np.where(Ca == 2, 1, 0)
    ncct_arrayy=ncct_array*reassigned_Ca
    ncct_arrayy=caw(ncct_arrayy)

    lumen_mask=copy.deepcopy(mask_image)#管腔
    lumen_mask[lumen_mask > 1] = 0
    lumen_mask = remove_small_volums(lumen_mask)
    diameter_per_lumen,area_per_lumen,_=diameter_area(lumen_mask,ncct_arrayy)

    total_mask = np.where(mask_image >0, 1, 0)#血管
    total_mask=remove_small_volums(total_mask)
    diameter_per_total,area_per_total,ncct_arrayyy=diameter_area(total_mask,ncct_arrayy)

    # per_index = diameter_per_lumen / diameter_per_total  # 钙化指数应该是钙化面积/(管腔+钙化)
    # per_index = np.ones(mask_image.shape[0]) * 0
    per_index = [diameter_per_lumen[i] / diameter_per_total[i] if diameter_per_total[i] != 0 else 0 for i in range(len(diameter_per_total))]
    per_index= np.array(per_index)
    per_index[per_index==0]=1
    per_index=1-per_index
    # per_index[per_index == 1] = 0
    per_index = np.clip(per_index, 0, 1)
    per_index[np.isnan(per_index)] = 0
    per_index[np.isinf(per_index)] = 0
    per_index = per_index.tolist()
    # per_index = [num for num in per_index if num != 0]

    lumen_mask=copy.deepcopy(mask_image)#管腔
    lumen_mask[lumen_mask > 2] = 1 #计算钙化斑块引起的狭窄
    lumen_mask[lumen_mask > 1] = 0
    #lumen_mask = remove_small_volums(lumen_mask)
    diameter_per_lumenc,area_per_lumenc,_=diameter_area(lumen_mask,ncct_arrayy)
    per_indexc = [diameter_per_lumenc[i] / diameter_per_total[i] if diameter_per_total[i] != 0 else 0 for i in range(len(diameter_per_total))]
    per_indexc= np.array(per_indexc)
    per_indexc[per_indexc==0]=1
    per_indexc=1-per_indexc
    # per_indexc[per_indexc == 1] = 0
    per_indexc = np.clip(per_indexc, 0, 1)
    per_indexc[np.isnan(per_indexc)] = 0
    per_indexc[np.isinf(per_indexc)] = 0
    per_indexc = per_indexc.tolist()

    lumen_mask=copy.deepcopy(mask_image)#管腔
    lumen_mask[lumen_mask > 2] = 0 #计算软斑块引起的狭窄
    lumen_mask[lumen_mask > 0] = 1
    lumen_mask = remove_small_volums(lumen_mask)
    diameter_per_lumens,area_per_lumens,_=diameter_area(lumen_mask,ncct_arrayy)

    # per_index = diameter_per_lumen / diameter_per_total  # 钙化指数应该是钙化面积/(管腔+钙化)
    # per_index = np.ones(mask_image.shape[0]) * 0
    per_indexs = [diameter_per_lumens[i] / diameter_per_total[i] if diameter_per_total[i] != 0 else 0 for i in range(len(diameter_per_total))]
    per_indexs= np.array(per_indexs)
    per_indexs[per_indexs==0]=1
    per_indexs=1-per_indexs
    # per_indexs[per_indexs == 1] = 0
    per_indexs = np.clip(per_indexs, 0, 1)
    per_indexs[np.isnan(per_indexs)] = 0
    per_indexs[np.isinf(per_indexs)] = 0
    per_indexs = per_indexs.tolist()

    # file=path.replace(".nii.gz",'.h5')
    h5_path_save = path_save.replace(".nii.gz", ".h5")
    if os.path.exists(h5_path_save):
        os.remove(h5_path_save)
    with h5py.File(h5_path_save, 'w') as f:# 创建一个dataset
        f.create_dataset('diameter_per_lumen', data=diameter_per_lumen)#管腔
        f.create_dataset('diameter_per_lumenc', data=diameter_per_lumenc)#管腔
        f.create_dataset('diameter_per_lumens', data=diameter_per_lumens)#管腔
        f.create_dataset('area_per_lumen', data=area_per_lumen)
        f.create_dataset('area_per_lumecn', data=area_per_lumenc)
        f.create_dataset('area_per_lumens', data=area_per_lumens)
        f.create_dataset('diameter_per_total', data=diameter_per_total)#整体
        f.create_dataset('area_per_total', data=area_per_total)
        f.create_dataset('per_index', data=per_index)#钙化 ncct_arrayyy
        f.create_dataset('per_indexc', data=per_indexc)#钙化 ncct_arrayyy
        f.create_dataset('per_indexs', data=per_indexs)  # 软斑块引起的狭窄
        f.create_dataset('ncct_arrayyy', data=ncct_arrayyy)

def calcium_alert_index(diameter_per_total, calcium_per_index_mea, calcium_per_index):
    # 指定一个阈值
    length=len(diameter_per_total)
    flag=0
    threshold = 13 #int(0.67*13）=8.71   #initial screening
    above_threshold = diameter_per_total > threshold
    greater_indices = np.where(above_threshold)[0]
    if greater_indices.size > 0:
        start_index = greater_indices[0]
        end_index = greater_indices[-1]  # 注意这里需要加1,因为end_index是要包含在内的
        if length-end_index>48:  ##further screening
            end_index=length-48  #remove the small branch vessels
        if end_index-start_index<256:
            start_index = end_index - 256 #小于则补充全
            if start_index<0:
                start_index=0
        else:
            flag=1
        ####need change
        # start_index = start_index + int((end_index-start_index)*0.40)#a=胸主动脉  注意数据是从腹部到胸部
        # end_index = start_index + int((end_index - start_index) * 0.40)  # b=腹主动脉。腹主动脉为雄主动脉 1.2-1.5

        # 将原数组中不在指定区间的数值赋值为0
        calcium_per_index_mea[:start_index] = 0
        calcium_per_index_mea[end_index:] = 0
        cai_mea = calcium_per_index_mea.max()#np.percentile(calcium_per_index_mea, 75)
        sorted_arr = np.sort(calcium_per_index_mea)[::-1]
        # top_mea = np.mean(sorted_arr[int(len(sorted_arr) * 0.95):])#[:int(len(sorted_arr) * 0.25)])
        top_mea = np.mean(sorted_arr[1:2])  # 计算前5个最大值的平均值

        calcium_per_index[:start_index] = 0
        calcium_per_index[end_index:] = 0
        cai_gd = calcium_per_index.max()#np.percentile(calcium_per_index, 75)
        sorted_arr = np.sort(calcium_per_index)[::-1]
        # top_gd = np.mean(sorted_arr[int(len(sorted_arr) * 0.95):])#[:int(len(sorted_arr) * 0.25)])
        top_gd = np.mean(sorted_arr[1:2])  # 计算前5个最大值的平均值
        # return cai_mea, cai_gd,start_index,end_index
        return top_mea, top_gd,start_index,end_index,flag

        # indices = np.nonzero(calcium_per_index_mea)
        # if len(indices[0])<2:
        #     cai_mea=0
        #     top_mea=0
        # else:
        #     calcium_per_index_mea = calcium_per_index_mea[indices]
        #     cai_mea = np.percentile(calcium_per_index_mea, 75)
        #     sorted_arr = np.sort(calcium_per_index_mea)[::-1]
        #     top_mea = np.mean(sorted_arr[:int(len(sorted_arr) * 0.25)])


        # indices = np.nonzero(calcium_per_index)
        # if len(indices[0])<20:
        #     cai_gd=0
        #     top_gd=0
        # else:
        #     calcium_per_index = calcium_per_index[indices]
        #     cai_gd = np.percentile(calcium_per_index, 75)
        #     sorted_arr = np.sort(calcium_per_index)[::-1]
        #     top_gd = np.mean(sorted_arr[:int(len(sorted_arr) * 0.25)])
        # return cai_mea,top_mea, cai_gd,top_gd
    else:
        return 0,0,0,0,flag

def dis_calssess(diameter_per_total, per_index_mea,per_index):
    sten_mea, sten_gd, start_index, end_index, flag = calcium_alert_index(diameter_per_total, per_index_mea, per_index)  # 注意注释掉分段
    # sten_mea, sten_gd, start_index, end_index = calcium_alert_index(diameter_per_lumen, per_index_mea, per_index)

    if sten_gd >= 0.0 and sten_gd < 0.05:
        stenosis_gd = 1
        # if su2[start_index:end_index].sum()>50:
        #     stenosis_gd=2
    elif sten_gd >= 0.05 and sten_gd <= 0.25:
        stenosis_gd = 2
    # if sten_gd >= 0.5 and sten_gd < 0.7:
    #     stenosis_gd = 3
    elif sten_gd > 0.25:
        stenosis_gd = 3
    return stenosis_gd

if __name__ == '__main__':
    # predict a bunch of files
    predictor = nnUNetPredictor(
        tile_step_size=0.5,#0.5
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
        )
    #nnUNetTrainer_p2_x2
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset301_Aorta/nnUNetTrainer__nnUNetPlans__3d_fullres'),
        use_folds=(0, ),
        checkpoint_name='checkpoint_best.pth',#checkpoint_best  checkpoint_final
    )

    f = open("/media/bit301/data/yml/project/python39/p2/Aorta_net/data/t_test_list.txt")  # hnnk test.txt
    output_file="/media/bit301/data/yml/project/python39/p2/Aorta_net/data/t_test_results.txt"
    path_list = []
    ij=0
    for line in f.readlines():#tile_step_size=0.75较好处理官腔错位问题
            start_time1 = time.time()
            path=line.split('\n')[0]
            img, props = SimpleITKIO().read_images([path])
            ret = predictor.predict_single_npy_array(img, props, None, None, False)

            # out_put=path.replace("external","t_test/external")#lo:97  x1:54  xlo2:10
            # out_put = out_put.split("0.nii.gz")[0]
            # file_path = os.path.join(out_put, "2.nii.gz")
            # mask = sitk.ReadImage(file_path, sitk.sitkInt16)
            # ret = sitk.GetArrayFromImage(mask)

            ncct=np.squeeze(img)
            ret=postpossess(ncct,ret)
            end_time1 = time.time()
            elapsed_time1 = end_time1 - start_time1# 计算运行时间

            start_time2 = time.time()
            #unet MedNeXtl MedNeXtx1c MedNeXtx1
            out_put=path.replace("external","t_test/external")#lo:97  x1:54  xlo2:10
            out_put = out_put.split("0.nii.gz")[0]
            if not os.path.isdir(out_put):
                os.makedirs(out_put)
            file_path = os.path.join(out_put, "2.nii.gz")
            mask = sitk.GetImageFromArray(ret.astype(np.int16))
            sitk.WriteImage(mask, file_path)
            ret = get_longest_3d_mask(ret)  #get_longest_3d_mask   remove_regions
            caculate_index(ret,ncct,file_path)
            end_time2 = time.time()
            elapsed_time2 = end_time2 - start_time2

            start_time3 = time.time()
            h5_path_save = file_path.replace(".nii.gz", ".h5")
            with h5py.File(h5_path_save, 'r') as f_gd:  # 评估四分位数据
                # diameter_per_total = f_gd['diameter_per_lumen'][:]##diameter_per_lumen diameter_per_total
                # area_per_lumen = f_gd['area_per_lumen'][:]
                diameter_per_total = f_gd['diameter_per_total'][:]
                # area_per_total = f_gd['area_per_total'][:]
                per_index = f_gd['per_index'][:]
                per_indexc = f_gd['per_indexc'][:]
                per_indexs = f_gd['per_indexs'][:]
            class_type=dis_calssess(diameter_per_total, per_index, per_index)
            class_typec = dis_calssess(diameter_per_total, per_indexc, per_indexc)
            class_types = dis_calssess(diameter_per_total, per_indexs, per_indexs)
            end_time3 = time.time()
            elapsed_time3 = end_time3 - start_time3
            with open(output_file, 'a') as file:
                file.write(path+"\n")
                file.write(f"volume size: {ncct.shape}\n")
                file.write(f"inference time(s): {elapsed_time1}\n")
                file.write(f"geometric caculate time(s): {elapsed_time2}\n")
                file.write(f"evaluation time(s): {elapsed_time3}\n")
                file.write("\n")

            ij = ij + 1
            if ij % 10 == 0:
                print('numbers:', ij)
    print('finished:!')
