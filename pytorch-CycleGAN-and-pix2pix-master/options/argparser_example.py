import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('square', help='square the input number', type=int)
# args  = parser.parse_args()
# print(args.square**2)


parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")