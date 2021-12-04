#import ihm.reader
import sys
import os
import glob
import shutil
import subprocess
import unittest
#import RMF


MODELLER_TEST_DIR = os.path.abspath("../comparative_modeling/s1-1_modeller_test")


class Tests(unittest.TestCase):
    
    def test_modeller(self):
        """
        Test if MODELLER runs succesfully (and in parallel) to get the 
        vanilla and the loop-refined model of s1-1. 
        """
        currdir = os.getcwd()
        
        # run the modeling script
        os.chdir(MODELLER_TEST_DIR)
        p = subprocess.check_call(["python", "modeling.py"])
        
        # check that 2 models before and 4 models after loop-refinement) have been computed
        models_1 = glob.glob(os.path.join(MODELLER_TEST_DIR, "s1-1.B9*.pdb"))
        assert sum([os.path.isfile(fn) for fn in models_1]) == 2

        models_2 = glob.glob(os.path.join(MODELLER_TEST_DIR, "s1-1.BL0*.pdb"))
        assert sum([os.path.isfile(fn) for fn in models_2]) == 4
        
        # check that top scoring files have been computed
        top_scoring_models = [os.path.join(MODELLER_TEST_DIR, "top_scoring_model.pdb"),
                              os.path.join(MODELLER_TEST_DIR, "top_scoring_loop_refined_model.pdb")]

        assert all([os.path.isfile(fn) for fn in top_scoring_models])
        
        # remove all output created till now
        del_files = glob.glob(os.path.join(MODELLER_TEST_DIR, "s1-1.*")) + \
                    glob.glob(os.path.join(MODELLER_TEST_DIR, "modeling.slave*")) + \
                    top_scoring_models
                    
        for fn in del_files:
            if os.path.isfile(fn):
                os.remove(fn)
                
        # switch to current dir
        os.chdir(currdir)
    
    
    def _test_integrative_model(self):
        """
        Test stuff
        """
        currdir = os.getcwd()
        
        # run the modeling script from a tmp dir
        outdir = os.path.abspath("tmp_run")
        os.makedirs(outdir, exist_ok=True)
        os.chdir(outdir)
        
        script = os.path.join(INTEGRATIVE_MODELING_DIR, "modeling.py")
        p = subprocess.check_call(["python", script, "-d", DATADIR, "-t"])
        
        rmf_fn_1 = os.path.join(outdir, "output_warmup", "rmfs", "0.rmf3")
        assert os.path.isfile(rmf_fn_1)
        rh1 = RMF.open_rmf_file_read_only(rmf_fn_1)
        assert rh1.get_number_of_frames() == N_WARMUP_MODELS
        
        rmf_fn_2 = os.path.join(outdir, "output", "rmfs", "0.rmf3")
        assert os.path.isfile(rmf_fn_2)
        rh2 = RMF.open_rmf_file_read_only(rmf_fn_2)
        assert rh2.get_number_of_frames() == N_PRODUCTION_MODELS
        
        # switch to current dir
        os.chdir(currdir)
        
        # delete the tmp output dir
        shutil.rmtree(outdir, ignore_errors=True)    


if __name__ == "__main__":
    unittest.main()
