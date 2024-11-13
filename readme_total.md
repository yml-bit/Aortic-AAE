1-install nnunet:details can be see[https://github.com/MIC-DKFZ/nnUNet]
2-data process:use test/data_process.py
	(1)resample and convert DICOM data to nii.gz file
	(2)make annotation and process,including mask Silhouette and multi-category mask merge
3-nnunet-v2 training:
	(1)set correct path：nnUNet/nnunetv2/path.py #
	(2)data prepare：nnUNet/nnunetv2/dataset_conversion/Datasets301_Aorta.py  #read the code carefully and figure out format of the output file
	(3)data checking and training preparation:execute a command "nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity"  under the path nnunet-v2
	(3)training:rename training(such as nnUNetTrainer_p2_AAE.py) file to nnUNetTrainer.py  #need to synchronize revision output path
4-inference: 
	(1)nnUNet/nnunetv2/inference/predict_P2_s.py (unet，Aortic-AAE and medxnet) or predict_P2_ns.py（others）  #mask sure the input and output file is right. Aortic-AAE pre-trained model[https://zenodo.org/records/14126680]
	(2) reprocess(just for Aortic-AAE): use test/data_process.py run postpossess function
5-segmentation evalution：overal_test.py
6-geometric cacaulate statis and make display chart:aortic_index_cs.py
7-instance total time cost test:nnUNet/nnunetv2/inference/predict_P2_t.py  pretrained model,test results, and the sample_data for instance time test can be download from [https://zenodo.org/records/14126680]

