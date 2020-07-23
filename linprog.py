# encode: utf8
from __future__ import print_function

import sys
from functools import reduce

from scipy import linalg

from utils import *

class Optimum(object):
    def __init__(self, **argv):
        # Status: solved 0, illegal -1, unbounded -2, max_iter -3
        self.status = argv.get("status", 0)
        self.z_opt = argv.get("z_opt")
        self.x_opt = argv.get("x_opt")
        self.lmbd_opt = argv.get("lmbd_opt")
        self.basis = argv.get("basis")
        self.x_basis = argv.get("x_basis")
        self.lu_basis = argv.get("lu_basis")
        self.inv_basis = argv.get("inv_basis")
        self.num_iter = argv.get("num_iter", 0)
        self.num_col = len(self.x_opt) if self.x_opt is not None else 0
        self.num_row = len(self.basis) if self.basis is not None else 0

    def __str__(self):
        return "\noptimum = %s\nnum_iter = %s\nx_opt = %s\nbasis = %s\n" % (
            self.z_opt, self.num_iter, str(self.x_opt), str(self.basis))


def form_standard(c, A_eq=None, b_eq=None, A_ub=None, b_ub=None, lower={}, upper={}, **argv):
    """
    Convert the linear program to standard form
    Input:
    Return:
        Success: (c_tot, A_tot, b_tot)
        Fail:
        -1: illegal
    """
    # init
    debug = argv.get("debug", False)
    # check 
    if (A_eq is not None and b_eq is None) \
            or (A_ub is not None and b_ub is None):
        sys.stder.write("Problme illegal\n")
        return -1
    # Problem size
    num_var = len(c)
    num_eq = A_eq.shape[0] if A_eq is not None else 0
    num_ub = A_ub.shape[0] if A_ub is not None else 0
    num_lower = len(lower)
    num_upper = len(upper)
    num_slack = num_ub + num_lower + num_upper
    row_tot = num_eq + num_slack
    col_tot = num_var + num_slack
    A_var = []
    b_tot = []
    ### equality
    if A_eq is not None:
        A_var.append(A_eq)
        b_tot.append(b_eq)
    ### inequality
    if A_ub is not None:
        A_var.append(A_ub)
        b_tot.append(b_ub)
    ### lower bounds
    eye_var = np.eye(num_var)
    if len(lower) > 0:
        lower_idx = sorted(lower.keys())
        b0 = -np.array([lower[i] for i in lower_idx])
        A0 = -eye_var.take(lower_idx, axis=0)
        A_var.append(A0)
        b_tot.append(b0)
    ### upper bounds
    eye_var = np.eye(num_var)
    if len(upper) > 0:
        upper_idx = sorted(upper.keys())
        b0 = np.array([upper[i] for i in upper_idx])
        A0 = eye_var.take(upper_idx, axis=0)
        A_var.append(A0)
        b_tot.append(b0)
    b_tot = np.concatenate(b_tot)
    A_var = np.concatenate(A_var)
    A_slack = np.concatenate((np.zeros((num_eq, num_slack)), np.eye(num_slack)))
    A_tot = np.concatenate((A_var, A_slack), axis=1)
    c_tot = np.concatenate((c, np.zeros(num_slack)))
    return c_tot, A_tot, b_tot


def find_null_variable(basis, A, x_basis, **argv):
    """ Find null variable to reduce equality.
    Returns:
        null_row: row of zero value
        null_val: index of null variable
    """
    lu_basis = argv.get("lu_basis")
    row, col = A.shape
    is_slack = lambda c: c >= col
    nonbasis = [i for i in range(col) if i not in basis]
    D = A.take(nonbasis, axis=1)
    null_row = []
    null_var = []
    for rid in range(len(x_basis)):
        if not is_zero(x_basis[rid]):
            continue
        var = basis[rid]
        inv_basis_row = linalg.lu_solve(lu_basis, get_unit_vector(row, rid), trans=1)
        y_row = inv_basis_row.dot(D)
        idx_nonzero = [i for i in range(len(y_row)) if not is_zero(y_row[i])]
        y_nonzero = y_row[idx_nonzero]
        var_nonzero = take_index(nonbasis, idx_nonzero)
        if is_slack(var) and (is_pos_all(y_nonzero) or is_neg_all(y_nonzero)):
            null_row.append(rid)
            null_var.append(var_nonzero)
        elif not is_slack(var) and is_pos_all(y_nonzero):
            var_nonzero.append(var)
            null_row.append(rid)
            null_var.append(var_nonzero)
    return null_row, null_var


def reduce_equation(null_row, null_var, c, A, b, basis):
    row, col = A.shape
    null_col = reduce(lambda x, y: x + y, null_var)
    row_res = [i for i in range(row) if i not in null_row]
    col_res = [i for i in range(col) if i not in null_col]
    c_res = c[col_res]
    b_res = b[row_res]
    A_res = A.take(row_res, axis=0)
    A_res = A_res.take(col_res, axis=1)
    basis_res = take_index(basis, row_res)
    return c_res, A_res, b_res, basis_res


def init_basis_primal(A, b, **argv):
    """
    Solve Artificial Linear Programming
        min 1*s 
        s.t s + A*x = b,
            s, x >= 0
    Input:
        A: equation constraint
        b: equation constraint
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: (basis, x, lambda)
        fail:
        -1: invalid
        -2: infeasible, the minimum is not zero
    """
    eps = argv.get("eps", 1e-10)
    row, col = A.shape
    cp = np.concatenate((np.zeros(col), np.ones(row)))
    Ap = np.concatenate((A, np.eye(row)), axis=1)
    basis = range(col, col + row)
    ret = simplex_revised(cp, Ap, b, basis, ret_lu=True)
    if type(ret) == int:
        sys.stderr.write("Problem invalid\n")
        return -1
    if not is_zero(ret.z_opt, eps):
        sys.stderr.write("Problem infeasible\n")
        return -2
    return ret


def check_basis_slack(basis, A, **argv):
    """ Whether the basis has slack variables 
    Input:
        basis: index of basic solution
        A: constraint matrix
        replace: whether or not replace slack variables 
    Return
        0: not slack 
        1: has slack 
    """
    row, col = A.shape
    idx_slack = [i for i in range(len(basis)) if basis[i] >= col]
    if len(idx_slack) == 0:
        return 0
    nonbasis = [i for i in range(col) if i not in basis]
    ## Replace slack with first non-basis
    ## TODO whether the basis is singular
    replace = argv.get("replace", True)
    if replace:
        j = 0
        for i in idx_slack:
            basis[i] = nonbasis[j]
            j += 1
    ## TODO Extend A and c
    return 1


def linprog_primal(c, A, b, **argv):
    """
    Solve Linear Programming in standard form
        min c*x 
        s.t A*x = b,
            x >= 0
    Input:
        c: object vector
        A: equation constraint
        b: equation constraint
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: (basis, x, lambda)
        fail:
        -1: illegal
        -2: unbounded
        -3: infeasible
    """
    # Init
    eps = argv.get("eps", 1e-16)
    debug = argv.get("debug", False)
    is_neg = lambda x: x < -eps
    # size
    row, col = A.shape
    if debug:
        print("\nProblem size row %s col %s" % (row, col))
    # Make sure b >= 0
    for i in range(row):
        if is_neg(b[i]):
            b[i] = -b[i]
            A[i] = -A[i]
            # Init basic solution
    ret0 = init_basis_primal(A, b)
    if type(ret0) == int:
        sys.stderr.write("Problem infeasible\n")
        return -3
    basis = ret0.basis
    x0 = ret0.x_basis
    if debug:
        print("\nBasic Problem solved")
        print("basis\t%s" % str(basis))
        print("x0\t%s" % str(x0))
    null_row, null_var = find_null_variable(basis, A, x0, lu_basis=ret0.lu_basis)
    if len(null_row) != 0:
        sys.stderr.write("Reduce enable null_row %s null_var %s\n" % (str(null_row), str(null_var)))
        c, A, b, basis = reduce_equation(null_row, null_var, c, A, b, basis)
    check_basis_slack(basis, A)
    # Solve LP
    opt = simplex_revised(c, A, b, basis, debug=debug)
    if type(opt) == int:
        sys.stderr.write("Problem unsolved\n")
        return opt
    if debug:
        print("\nPrimal Problem solved")
        print("z_opt\t%s" % opt.z_opt)
        print("x_opt\t%s" % str(opt.x_opt))
    return opt


def linprog(c, **argv):
    c_tot, A_tot, b_tot = form_standard(c, **argv)
    return linprog_primal(c_tot, A_tot, b_tot, **argv)

def align_basis(x_b, basis, dim):
    x0 = np.zeros(dim)
    for i in range(len(basis)):
        x0[basis[i]] = x_b[i]
    return x0


def check_size(c, A, b, basis):
    row, col = A.shape
    if row != len(b) or col != len(c) or row != len(basis):
        return False
    else:
        return True


def simplex_revised(c, A, b, basis, **argv):
    """
    Revised simplex for Linear Programming
        min c*x 
        s.t A*x = b,
            x >= 0
    Input:
        c: object vector
        A: equation constraint
        b: equation constraint
        basis: index of basis
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: (basis, x, lambda)
        fail:
        -1: illegal
        -2: unbounded
        -3: unsolved
    """
    # argument
    eps = argv.get("eps", 1e-16)
    max_iter = argv.get("max_iter", 100)
    debug = argv.get("debug", False)
    ret_lu = argv.get("ret_lu", False)
    is_neg = lambda x: x < -eps
    is_pos = lambda x: x > eps
    is_zero = lambda x: x <= eps and x >= -eps
    # check problem 
    row, col = A.shape
    if not check_size(c, A, b, basis):
        info = "Size illegal c:%s A:%s,%s b:%s, basis:%s\n" % (len(c), row, col, len(b), len(basis))
        sys.stderr.write(info)
        return -1
    # check primal feasible
    if any(is_neg(i) for i in b):
        sys.stderr.write("Basis illegal b:%s" % str(b))
        return -1

    basis = list(basis)
    # iteration
    for itr in range(max_iter):
        nonbasis = [i for i in range(col) if i not in basis]
        B = A.take(basis, axis=1)
        c_b = c[basis]
        D = A.take(nonbasis, axis=1)
        c_d = c[nonbasis]
        # solve system B
        lu_p = linalg.lu_factor(B)
        x_b = linalg.lu_solve(lu_p, b)
        lmbd = linalg.lu_solve(lu_p, c_b, trans=1)
        r_d = c_d - lmbd.dot(D)
        z0 = np.dot(x_b, c_b)
        if debug:
            print("\nIteration %d" % itr)
            print("z\t%s" % z0)
            print("basis\t%s" % str(basis))
            print("x_b\t%s" % str(x_b))
            print("lambda\t%s" % str(lmbd))
            print("r_d\t%s" % str(r_d))
        # check reduced cost
        neg_ind = [i for i in range(len(r_d)) if is_neg(r_d[i])]
        if len(neg_ind) == 0:
            sys.stderr.write("Problem solved\n")
            x_opt = align_basis(x_b, basis, col)
            opt = Optimum(z_opt=z0, x_opt=x_opt, lmbd_opt=lmbd, basis=basis, x_basis=x_b, num_iter=itr)
            if ret_lu:
                opt.lu_basis = lu_p
            return opt
        ind_new = nonbasis[neg_ind[0]]
        # pivot
        a_q = A.take(ind_new, axis=1)
        y_q = linalg.lu_solve(lu_p, a_q)
        pos_ind = [i for i in range(len(y_q)) if is_pos(y_q[i])]
        if len(pos_ind) == 0:
            sys.stderr.write("Problem unbounded\n")
            return -2
        ratio = [x_b[i] / y_q[i] for i in pos_ind]
        min_ind = np.argmin(ratio)
        out = pos_ind[min_ind]
        ind_out = basis[out]
        basis[out] = ind_new
        if debug:
            print("y_q\t%s" % str(y_q))
            print("basis in %s out %s" % (ind_new, ind_out))
    sys.stderr.write("Iteration exceed %s\n" % max_iter)
    sys.stderr.write("Current optimum %s\n" % z0)
    return -3


def simplex_dual(c, A, b, basis, **argv):
    """
    Dual simplex for Linear Programming
        min c*x 
        s.t A*x = b,
            x >= 0
    Input:
        c: object vector
        A: equation constraint
        b: equation constraint
        basis: index of basis
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: (basis, x, lambda)
        fail:
        -1: illegal
        -2: unbounded
        -3: unsolved
    """
    # init 
    eps = argv.get("eps", 1e-16)
    max_iter = argv.get("max_iter", 100)
    debug = argv.get("debug", False)
    ret_lu = argv.get("ret_lu", False)
    is_neg = lambda x: x < -eps
    is_pos = lambda x: x > eps
    is_zero = lambda x: x <= eps and x >= -eps

    # check problem 
    row, col = A.shape
    if not check_size(c, A, b, basis):
        info = "Size illegal c:%s A:%s,%s b:%s, basis:%s\n" % (len(c), row, col, len(b), len(basis))
        sys.stderr.write(info)
        return -1

    basis = list(basis)
    # iteration
    for itr in range(max_iter):
        nonbasis = [i for i in range(col) if i not in basis]
        B = A.take(basis, axis=1)
        c_b = c[basis]
        D = A.take(nonbasis, axis=1)
        c_d = c[nonbasis]
        # solve system B
        lu_p = linalg.lu_factor(B)
        x_b = linalg.lu_solve(lu_p, b)
        lmbd = linalg.lu_solve(lu_p, c_b, trans=1)
        r_d = c_d - lmbd.dot(D)
        z0 = np.dot(x_b, c_b)
        # check dual feasible
        if any(is_neg(i) for i in r_d):
            sys.stderr.write("Dual infeasible r_d:%s\n" % str(r_d))
            return -1
        if debug:
            print("\nIteration %d" % itr)
            print("z\t%s" % z0)
            print("basis\t%s" % str(basis))
            print("x_b\t%s" % str(x_b))
            print("lambda\t%s" % str(lmbd))
            print("r_d\t%s" % str(r_d))
        # check x_b
        neg_ind = [i for i in range(len(x_b)) if is_neg(x_b[i])]
        if len(neg_ind) == 0:
            sys.stderr.write("Problem solved\n")
            x_opt = align_basis(x_b, basis, col)
            opt = Optimum(z_opt=z0, x_opt=x_opt, lmbd_opt=lmbd, basis=basis, x_basis=x_b, num_iter=itr)
            if ret_lu:
                opt.lu_basis = lu_p
            return opt
        ind_neg = neg_ind[0]
        ind_out = basis[ind_neg]
        # pivot
        e_q = np.zeros(row)
        e_q[ind_neg] = 1
        u_q = linalg.lu_solve(lu_p, e_q, trans=1)
        y_q = D.T.dot(u_q)
        y_neg = [i for i in range(len(y_q)) if is_neg(y_q[i])]
        if len(y_neg) == 0:
            sys.stderr.write("Problem unbounded\n")
            return -2
        ratio = [r_d[i] / -y_q[i] for i in y_neg]
        min_ind = np.argmin(ratio)
        ind_new = nonbasis[y_neg[min_ind]]
        basis[ind_neg] = ind_new
        if debug:
            print("y_q\t%s" % str(y_q))
            print("basis in %s out %s" % (ind_new, ind_out))
    sys.stderr.write("Iteration exceed %s\n" % max_iter)
    sys.stderr.write("Current optimum %s\n" % z0)
    return -3