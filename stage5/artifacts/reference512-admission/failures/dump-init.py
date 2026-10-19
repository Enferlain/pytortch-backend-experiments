import faulthandler
with open('/home/imi/Projects/pytortch-backend-experiments/stage5/artifacts/reference512-admission/failures/init-stack.log','a') as log:
    faulthandler.dump_traceback(file=log, all_threads=True)
