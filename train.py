# import utils.imutils as ut
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', help='path to directory with CT scans',
                required=True
                )
ap.add_argument('-m', '--model', help='path to save output model')

args = ap.parse_args()
