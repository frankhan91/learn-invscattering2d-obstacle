import os
import numpy as np
import argparse
import scipy.io
import logging
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./data/star3_kh10_n48_100/test", type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    inverse_dir = os.path.join(args.model_dir, "inverse")
    data_dirs = os.listdir(inverse_dir)
    l = len(data_dirs)
    logger.info("The number of inverse results is{:3}".format(l))
    err_l2 = np.zeros(l)
    err_l2_refined = np.zeros(l)-1
    err_Chamfer0 = np.zeros(l)
    err_Chamfer = np.zeros(l)
    for index, one_dir in enumerate(data_dirs):
        inverse_result = scipy.io.loadmat(os.path.join(inverse_dir,one_dir))
        err_l2[index] = inverse_result["err_l2"]
        if "err_l2_refined" in inverse_result:
            err_l2_refined[index] = inverse_result["err_l2_refined"]
        err_Chamfer0[index] = inverse_result["err_Chamfer"][0][0]
        err_Chamfer[index] = inverse_result["err_Chamfer"][0][1]
    logger.info("Mean of the L2 error {:.6f}".format(np.mean(err_l2)))
    logger.info("Mean of refined L2 error {:.10f} (only for star specific algorithm, otherwise -1)".format(np.mean(err_l2_refined)))
    logger.info("Mean of predicted Chamfer error {:.6f}".format(np.mean(err_Chamfer0)))
    logger.info("Mean of refined Chamfer error {:.10f}".format(np.mean(err_Chamfer)))

if __name__ == '__main__':
    main()