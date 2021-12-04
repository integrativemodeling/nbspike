import shutil
import numpy as np

from modeller import *
from modeller.automodel import *
from modeller.parallel import Job, LocalWorker

from nb_model import NanobodyModel, NanobodyLoopModel

N_MODELS = 20
N_LOOP_MODELS = 1000
NCORES = 20
SEQUENCE = "s2-40"

job = Job()
[job.append(LocalWorker()) for _ in range(NCORES)]

env = environ()
env.io.atom_files_directory = [".."]
env.io.hetatm = False

# --------------------
# COMPARATIVE MODELING
# --------------------
a = NanobodyModel(env,
                  alnfile="alignment.ali",
                  knowns="5iml.B",
                  sequence=SEQUENCE,
                  assess_methods=(assess.DOPE,
                                  assess.normalized_dope,
                                  assess.GA341))

a.starting_model = 1
a.ending_model = N_MODELS
a.md_level = refine.fast
a.use_parallel_job(job)
a.make()

# select top comparative model according to DOPE score
models = list(filter(lambda x: x["failure"] is None, a.outputs))
models = sorted(models, key=(lambda x: x["DOPE score"]))
best_model = models[0]
print("\nBest comparative model (without loop refinement) is %s, with DOPE score %.2f" % \
      (best_model["name"], best_model["DOPE score"]))
shutil.copyfile(best_model["name"], "top_scoring_model.pdb")
print("Refining loops in %s now..." % best_model["name"])


# -------------
# LOOP MODELING
# -------------
l = NanobodyLoopModel(env,
                  inimodel=best_model["name"],
                  sequence=SEQUENCE,
                  loop_assess_methods=assess.DOPE)

l.loop.starting_model = 1
l.loop.ending_model = N_LOOP_MODELS
l.loop.md_level = refine.fast
l.use_parallel_job(job)  # parallel execution
l.make()

# select best loop model according to DOPE score
loop_models = list(filter(lambda x: x["failure"] is None, l.loop.outputs))
loop_models = sorted(loop_models, key=(lambda x: x["DOPE score"]))
best_loop_model = loop_models[0]
print("Best model after loop refinement is %s, with DOPE score %.2f" % \
      (best_loop_model["name"], best_loop_model["DOPE score"]))
shutil.copyfile(best_loop_model["name"], "top_scoring_loop_refined_model.pdb")

