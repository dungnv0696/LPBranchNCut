# encode: utf8
import numpy as np

from branch_cut import branch_bound
from branch_cut import branch_cut
from linprog import form_standard
from linprog import linprog_primal


def test_branch_bound():
    print("\nTest Branch Bound")
    a, b, c = readInput()
    c = c*(-1)
    basis = [i for i in range(len(a))]
    ret = branch_bound(c, a, b, basis, debug=True, deep=False)
    nid, tree = ret
    node = tree[nid]
    print("z\t%s" % ([] if node.z_opt == [] else -node.z_opt))
    print("x_opt\t%s" % str(node.x_opt))

def test_branch_cut():
    print("\nTest Branch Cut")
    a, b, c = readInput()
    c = c*(-1)
    basis = [i for i in range(len(a))]
    ret = branch_cut(c, a, b, basis, debug=True, deep=False)
    nid, tree = ret
    node = tree[nid]
    print("z\t%s" % ([] if node.z_opt == [] else -node.z_opt))
    print("x_opt\t%s" % str(node.x_opt))

def readInput():
    print("Nhap ten file")
    fileName = input()
    f = open(fileName, 'r')
    constraint_num = int(f.readline())
    c = [float(ci) for ci in f.readline().split()]
    A = []
    for i in range (constraint_num):
        A.append([float(Aij) for Aij in f.readline().split()])
    b = [float(bi) for bi in f.readline().split()]

    return np.asarray(A), np.asarray(b), np.asarray(c)

if __name__ == "__main__":
    print("Chon option, Nhanh can = 0, Nhanh cat = 1")
    option = input()
    if(option == "0"):
        test_branch_bound()
    else:
        test_branch_cut()
