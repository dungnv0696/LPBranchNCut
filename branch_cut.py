# encode: utf8
from __future__ import print_function

import sys

import numpy as np
import math
from scipy import linalg

from utils import *
from linprog import *

class Node(object):
    def __init__(self, nid, pid=0, **argv):
        self.nid = nid
        self.pid = pid
        self.children = set()
        self.branch_type = argv.get("branch_type", 0)
        self.lower = dict(argv.get("lower", {}))
        self.upper = dict(argv.get("upper", {}))
        self.cut_set = argv.get("cut_set", set())
        self.num_var = argv.get("num_var", 0)
        self.num_slack = len(self.lower) + len(self.upper) + len(self.cut_set)
        self.basis = []
        self.basis_raw = argv.get("basis_raw", [])
        self.x_opt = []
        self.z_opt = []
        self.lmbd_opt = []
        self.cut_active = []
        self.is_solved = False
        self.is_int = False
        self.status = 0
        self.cover_cut = argv.get("cover_cut", False)
        self.row_idx = argv.get("row_idx", 0)
        self.num_constraint = argv.get("num_constraint", 0)

    def form_program(self, c_raw, A_raw, b_raw):
        ret = form_standard(c_raw, A_eq=A_raw, b_eq=b_raw, lower=self.lower, upper=self.upper)
        return ret

    def solve(self, c_raw, A_raw, b_raw, **argv):
        """ Solve subproblem
        Return:
            0: success
            -1: illegal 
            -2: problme unsolvable
        """
        # argv
        debug = argv.get("debug", False)
        # check
        if len(self.basis_raw) != len(b_raw):
            sys.stderr.write("Basic solution invalid\n")
            return -1
        # standard form
        ret_form = self.form_program(c_raw, A_raw, b_raw)
        if type(ret_form) == int:
            sys.stderr.write("Standard form invalid\n")
            return -1
        c_tot, A_tot, b_tot = ret_form
        num_var = len(c_raw)
        num_raw = len(b_raw)
        self.num_var = num_var
        basis_tot = self.basis_raw + list(range(num_var, num_var + self.num_slack))
        if(self.nid == 0 or self.cover_cut == True):
            opt = linprog_primal(c_tot, A_tot, b_tot)
        else:
            opt = simplex_dual(c_tot, A_tot, b_tot, basis_tot)
        if type(opt) == int:
            sys.stderr.write("Problem unsolvable\n")
            return -2
        self.is_solved = True
        self.basis = opt.basis
        self.x_opt = opt.x_opt
        self.lmbd_opt = opt.lmbd_opt
        self.z_opt = opt.z_opt
        self.basis_raw = self.basis[:num_raw]
        return 0

    @staticmethod
    def is_integer_solution(basis, x_opt, int_idx):
        for i in basis:
            if i >= len(x_opt):
                print("Integer fail %s %s %s" % (i, str(x_opt), str(basis)))
            if i in int_idx and not is_integer(x_opt[i]):
                return False
        return True

    def process(self, c_raw, A_raw, b_raw, int_idx, **argv):
        self.status = self.solve(c_raw, A_raw, b_raw, **argv)
        if self.is_solved:
            self.is_int = self.is_integer_solution(self.basis, self.x_opt, int_idx)


def branch_bound(c, A_eq, b_eq, basis, int_idx=None, **argv):
    """
    Branch algorithm for integer linear programming
    Return:
        node: the optimum node
        -1: illegal
    """
    ## argv
    max_iter = argv.get("max_iter", 100)
    max_iter_sub = argv.get("max_iter", 10000)
    debug = argv.get("debug", False)
    deep_first = argv.get("deep", True)
    num_var = len(c)
    if int_idx is None:
        int_idx = range(num_var)
    tree_dict = {}
    root_id = 0
    root = Node(0, basis_raw=basis)
    tree_dict[root_id] = root
    node_stack = [root_id]
    opt_val = 1e16
    opt_nid = 0
    active_cut_tot = {}
    for itr in range(max_iter):
        if len(node_stack) == 0:
            return opt_nid, tree_dict
        nid = node_stack.pop()
        if nid not in tree_dict:
            return -1
        node = tree_dict[nid]
        if debug:
            print("\nIteration %s" % itr)
            print("nid\t%s" % nid)
        ret = node.process(c, A_eq, b_eq, int_idx=int_idx, max_iter=max_iter_sub)
        if debug:
            print("Node")
            print("status\t%s" % node.status)
            print("z\t%s" % ([] if node.z_opt == [] else -node.z_opt))
            print("x\t%s" % node.x_opt)
        ## Pruning
        if node.status < 0:
            sys.stderr.write("SubProblem unsolvable\n")
            continue
        if node.z_opt >= opt_val:
            sys.stderr.write("SubProblem optimum %s over the best solution %s\n" % (-node.z_opt, -opt_val))
            continue
        if node.is_int:
            sys.stderr.write("SubProblem "+str(nid)+" has integer solution "+str(node.x_opt)+" , optimum "+str(-node.z_opt)+"\n")
            if node.z_opt < opt_val:
                opt_nid = nid
                opt_val = node.z_opt
            continue
        ## Branch
        cut_idx = 0
        var_idx = None
        b_val = None
        for i in node.basis:
            if not is_integer(node.x_opt[i]) and i in int_idx:
                var_idx = i
                b_val = node.x_opt[i]
                break
            cut_idx += 1
        ### upper bound
        upper = {}
        upper.update(node.upper)
        upper[var_idx] = np.floor(b_val)
        nid_ub = len(tree_dict)
        node_ub = Node(nid_ub, pid=nid, basis_raw=node.basis_raw, lower=node.lower, upper=upper)
        tree_dict[nid_ub] = node_ub
        ### lower bound
        lower = {}
        lower.update(node.lower)
        lower[var_idx] = np.ceil(b_val)
        nid_lb = len(tree_dict)
        node_lb = Node(nid_lb, pid=nid, basis_raw=node.basis_raw, lower=lower, upper=node.upper)
        tree_dict[nid_lb] = node_lb
        ### push stack
        if not deep_first:
            node_stack.append(nid_ub)
            node_stack.append(nid_lb)
        else:
            node_stack.append(nid_lb)
            node_stack.append(nid_ub)

        if debug:
            print("Branch")
            print("var\t%s" % var_idx)
            print("val\t%s" % b_val)
            print("stack\t%s" % str(node_stack))
    return opt_nid, tree_dict

def branch_cut(c, A_eq, b_eq, basis, int_idx=None, **argv):
    """
    Branch algorithm for integer linear programming
    Return:
        node: the optimum node
        -1: illegal
    """
    ## argv
    max_iter = argv.get("max_iter", 100)
    max_iter_sub = argv.get("max_iter", 10000)
    debug = argv.get("debug", False)
    deep_first = argv.get("deep", True)
    num_var = len(c)
    if int_idx is None:
        int_idx = range(num_var)
    tree_dict = {}
    root_id = 0
    root = Node(0, num_constraint = len(A_eq), row_idx = 0, basis_raw=basis)
    tree_dict[root_id] = root
    node_stack = [root_id]
    opt_val = 1e16
    opt_nid = 0
    active_cut_tot = {}
    for itr in range(100):
        if len(node_stack) == 0:
            return opt_nid, tree_dict
        nid = node_stack.pop()
        if nid not in tree_dict:
            return -1
        node = tree_dict[nid]
        if debug:
            print("\nIteration %s" % itr)
            print("nid\t%s" % nid)
        ret = node.process(c, A_eq, b_eq, int_idx=int_idx, max_iter=max_iter_sub)
        if debug:
            print("Node")
            print("status\t%s" % node.status)
            print("z\t%s" % ([] if node.z_opt == [] else -node.z_opt))
            print("x\t%s" % node.x_opt)
        ## Pruning
        if node.status < 0:
            sys.stderr.write("SubProblem unsolvable\n")
            continue
        if(math.isnan(node.z_opt)):
            continue
        if node.z_opt >= opt_val:
            sys.stderr.write("SubProblem optimum %s over the best solution %s\n" % (-node.z_opt, -opt_val))
            continue
        if node.is_int:
            sys.stderr.write("SubProblem %s has integer solution %s, optimum %s\n" % (nid, node.x_opt, -node.z_opt))
            if node.z_opt < opt_val:
                opt_nid = nid
                opt_val = node.z_opt
            continue

        ## Find Cover Cut
        cover_cut = False
        column_idx = None
        for j in range(node.row_idx, node.num_constraint):
            tong = 0
            for i in range(len(A_eq[j])):
                tong = tong + A_eq[j][i]
                if(tong > b_eq[j]):
                    cover_cut = True
                    row_idx = j
                    column_idx = i
                    break
            if(cover_cut == True):
                break

        if(cover_cut == True ):
            c = np.concatenate((c, [0]))
            A_eq = np.concatenate((A_eq, np.zeros((len(A_eq), 1))), axis=1)
            y_eq = np.concatenate((np.ones(column_idx+1), np.zeros(len(A_eq[0])-column_idx-2)))
            y_eq = np.concatenate((y_eq, [1])).reshape(1, len(y_eq)+1)
            A_eq = np.concatenate((A_eq, y_eq))
            b_eq = np.concatenate((b_eq, [column_idx+2]))
            node.basis_raw.append(len(A_eq))
            print(node.basis_raw)
            nid_cc = len(tree_dict)
            node_cc = Node(nid_cc, pid=nid,row_idx = row_idx, cover_cut = True, basis_raw=node.basis_raw)
            tree_dict[nid_cc] = node_cc
            node_stack.append(nid_cc)
        else:
            ## Branch
            cut_idx = 0
            var_idx = None
            b_val = None
            for i in node.basis:
                if not is_integer(node.x_opt[i]) and i in int_idx:
                    var_idx = i
                    b_val = node.x_opt[i]
                    break
                cut_idx += 1
            ### upper bound
            upper = {}
            upper.update(node.upper)
            upper[var_idx] = np.floor(b_val)
            nid_ub = len(tree_dict)
            node_ub = Node(nid_ub, pid=nid, basis_raw=node.basis_raw, lower=node.lower, upper=upper)
            tree_dict[nid_ub] = node_ub
            ### lower bound
            lower = {}
            lower.update(node.lower)
            lower[var_idx] = np.ceil(b_val)
            nid_lb = len(tree_dict)
            node_lb = Node(nid_lb, pid=nid, basis_raw=node.basis_raw, lower=lower, upper=node.upper)
            tree_dict[nid_lb] = node_lb
            ### push stack
            if not deep_first:
                node_stack.append(nid_ub)
                node_stack.append(nid_lb)
            else:
                node_stack.append(nid_lb)
                node_stack.append(nid_ub)

            if debug:
                print("Branch")
                print("var\t%s" % var_idx)
                print("val\t%s" % b_val)
                print("stack\t%s" % str(node_stack))
    return opt_nid, tree_dict