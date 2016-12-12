#this code was abridged from the cs_475 testing suite provided by Eric Bridgeford

from subprocess import Popen, PIPE
import time

def main():
	algorithms = ["pearson", "spearman", "mean_squared", "cosine"]
	for algorithm in algorithms:
		cmd_train = "python Code/runner.py --mode train --algorithm " + algorithm + " --model-file " + algorithm + ".model --data Data/ratings.csv"
		execute_cmd(cmd_train, algorithm)
	for algorithm in algorithms:
		for num_neighbors in range(2, 10):
			cmd_test = "python Code/runner.py --mode test --algorithm " + algorithm + " --model-file "  + algorithm + ".model --num-neighbors " + str(num_neighbors) + " --data Data/ratings.csv --predictions-file " + algorithm + ".predictions"
			start = time.time()
			execute_cmd(cmd_test, algorithm)
			run_time = time.time() - start
			cmd_cmp = "python Code/eval.py Data/ratings.csv "  + algorithm + ".predictions"
			(acc, err) = execute_cmd(cmd_cmp, algorithm)
			accuracy = (acc[:-1][acc.index(":"):][2:])
			print(algorithm + "," + str(num_neighbors) + "," + str(accuracy) + "," + str(run_time))

def execute_cmd(cmd, algorithm=None):
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    code = p.returncode
    if code:
        err_mssg = ""
        if algorithm is not None:
            err_mssg += "Your code did not work for algorithm: " + algorithm
        else:
            raise ValueError(err_mssg)
    return out, err

if __name__ == "__main__":
    main()