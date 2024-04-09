import os
import subprocess
import sys
import shutil
import time

from utils import EnterDir

class RunQubic:
    """
    Run Qubic using subprocess module
    """

    BIN = ("sci.x", "ctns.x", "post.x")

    def __init__(self, qubic_path: str, input_path) -> None:
        self.check_path(qubic_path, input_path)
        self.qubic_path = qubic_path
        self.input_path = input_path
    
    def run(self, input_file: str, integral_file: str) -> None:
        """
        Run sci.x, ctns.x using subprocess module
        """
        # Enter input-file dir
        with EnterDir(self.input_path):
            sys.stdout.write(f"Enter {self.input_path} dir: {time.ctime()}\n")
            sys.stdout.write(f"Qubic bin prefix: {self.qubic_path}\n")
            sys.stdout.flush()
            t0 = time.time_ns()

            # copy integral file
            command = f"grep 'integral_file' {input_file} -m -1"
            path = subprocess.check_output(command, shell=True, text=True).replace("\n","").split()[-1]
            path = os.path.join(self.input_path, path)
            shutil.copy(integral_file, path)
            
            # run sci.x
            sci = os.path.join(self.qubic_path, "sci.x")
            command = sci + " " + input_file
            sys.stdout.write(f"Run sci.x {input_file}\n")
            sys.stdout.flush()
            proc = subprocess.Popen(command, shell=True)
            proc.wait()

            # run ctns.x
            ctns = os.path.join(self.qubic_path, "ctns.x")
            command = ctns + " " + input_file
            sys.stdout.write(f"Run ctns.x {input_file}\n")
            sys.stdout.flush()
            proc = subprocess.Popen(command, shell=True)
            proc.wait()

            t1 = time.time_ns()
            sys.stdout.write(f"End SCI and CTNS calculation: {(t1-t0)/1.0E09:.3E} s\n")
        


    def check_path(self, qubic_path, input_path) ->None:
        """
        check Qubic absolute path
        """
        if not os.path.isdir(qubic_path):
            raise ValueError(f"Qubic bin Path: {qubic_path} dose not exits")
        
        if not os.path.isdir(input_path):
            raise ValueError(f"Input file Path: {input_path} dose not exits")

        for i in self.BIN:
            bin_paths = os.path.join(qubic_path, i)
            if not os.path.exists(bin_paths):
                raise FileNotFoundError(f"Bin file {bin_paths} does not exits")