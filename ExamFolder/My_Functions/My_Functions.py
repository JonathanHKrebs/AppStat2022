import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
from scipy import stats
import math 
import sympy as sp

class Error_Prop:
    def __init__(self, eq, sym_var, sym_err, vals, err_vals, corr):
        ''' Equation, symbolic variables, symbolic errors, values, error values, correlation matrix'''
        self.eq = eq
        self.sym_var = sym_var
        self.sym_err = sym_err
        self.vals = vals
        self.err_vals = err_vals
        self.corr = corr
        self.covar = self.corr.copy()
        for i in range(len(self.covar)):
            for j in range(len(self.covar)):
                self.covar[i][j] *= err_vals[i] * err_vals[j]
        self.derivatives = [self.eq.diff(var) for var in sym_var]
        self.functions = [sp.lambdify((*sym_var, *sym_err), derive) for derive in self.derivatives]
        self.diff_vals = [func(*vals, *err_vals) for func in self.functions]

    def get_propEq(self):
        propEq = 0
        for i in range(len(self.diff_vals)):
            for j in range(len(self.diff_vals)):
                propEq += self.derivatives[i] * self.derivatives[j] * self.corr[i][j]* self.sym_err[i] * self.sym_err[j]
        return sp.sqrt(propEq)

    def get_error(self):
        error = 0
        for i in range(len(self.diff_vals)):
            for j in range(len(self.diff_vals)):
                error += self.diff_vals[i] * self.diff_vals[j] * self.covar[i][j]
        return np.sqrt(error)
    
    def get_contributions(self):
        contributions = []
        for i in range(len(self.diff_vals)):
            contributions.append(self.diff_vals[i]**2 * self.covar[i][i])
        return contributions
    
    def update(self):
        self.covar = self.corr.copy()
        for i in range(len(self.covar)):
            for j in range(len(self.covar)):
                self.covar[i][j] *= self.err_vals[i] * self.err_vals[j]
        self.derivatives = [self.eq.diff(var) for var in self.sym_var]
        self.functions = [sp.lambdify((*self.sym_var, *self.sym_err), derive) for derive in self.derivatives]
        self.diff_vals = [func(*self.vals, *self.err_vals) for func in self.functions]        


def chi2_constant(c, x, xerr):
    chi2 = np.sum(((c - x)/xerr)**2)
    pval = stats.chi2.sf(chi2, len(x) - 1)
    return chi2, pval

def norm_gauss(x, mu, sigma):
    return stats.norm.pdf(x, mu, sigma)

def gauss(x, mu, sigma, N):
    return N*norm_gauss(x, mu, sigma)

def double_gauss(x, mu_1, sigma_1, mu_2, sigma_2, N, f):
    return N*(f*norm_gauss(x, mu_1, sigma_1) + (1-f)*norm_gauss(x, mu_2, sigma_2))

def linear(x, a, b):
    return a*x + b

def poly2(x, a, b, c):
    return a*x**2 + b*x + c

def poly3(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d
    
def create_histogram(data, n_bins, *args):
    ''' Data, number of bins'''
    counts, bin_edges = np.histogram(data, bins=n_bins, *args)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    bin_width = bin_edges[1]-bin_edges[0]
    return counts, bin_edges, bin_centers, bin_width

def normalization_constant(func, var, norm_const, range):
    ''' Function, variable, normalization constant, range'''
    integral = sp.integrate(func, (var, *range))
    val_const = norm_const/integral
    return val_const, integral

def transform_method(f, var, range, N_points, random):
    ''' Function, variable, range, number of points, random generator'''
    F = sp.integrate(f, (var, range[0], var))
    integral = sp.integrate(f, (var, *range))
    F = F/integral
    u = sp.symbols('u', positive = True)
    inverse = sp.solve(F - u, var)[0]
    inverse_func = sp.lambdify(u, inverse)
    us = random.uniform(size = N_points)
    return inverse_func(us)

def accept_reject(N_points, xrange, yrange, random, func, *args):
    ''' Number points, x range, y range, random generator, function, function arguments'''
    N_tries = 0
    accepted = []
    for i in range(N_points):
        while True:
            N_tries += 1
            x_test = random.uniform(*xrange)
            y_test = random.uniform(*yrange)
            if y_test < func(x_test, *args):
                break
        accepted.append(x_test)
    return accepted, N_tries

def Monte_Carlo_combi(N_points, random, func, gen_func, var, xrange, *args):
    ''' Number points, random generator, function, generating function, variable, range, function arguments'''
    gen_func_lambda = sp.lambdify(var, gen_func)
    F = sp.integrate(gen_func, (var, xrange[0], var))
    integral = sp.integrate(gen_func, (var, *xrange))
    F = F/integral
    u = sp.symbols('u', positive = True)
    inverse = sp.solve(F - u, var)[0]
    inverse_func = sp.lambdify(u, inverse)
    N_tries = 0
    accepted = []
    for i in range(N_points):
        while True:
            N_tries += 1
            x_test = inverse_func(random.uniform())
            y_test = random.uniform(0, gen_func_lambda(x_test))
            if y_test < func(x_test, *args):
                break
        accepted.append(x_test)
    return accepted, N_tries

def evaluate_chi2(minuit_obj, len_counts):
    ''' Minuit object, number of counts'''
    chi2 = minuit_obj.fval
    ndof = len_counts - len(minuit_obj.values[:])
    chi2_prob = stats.chi2.sf(chi2, ndof)
    return chi2, ndof, chi2_prob  

def ROC_curve(positive, negative, range, above = True):
    ''' Positive, negative, range, above = True'''
    TPR = []
    FPR = []
    if above:
        for i in range:
            TPR.append((positive > i).sum()/len(positive))
            FPR.append((negative > i).sum()/len(negative))
    else:
        for i in range:
            TPR.append((positive < i).sum()/len(positive))
            FPR.append((negative < i).sum()/len(negative))
    return np.array(TPR), np.array(FPR)

def ROC_Fit(fit_positive, fit_negative):
    ''' Fit positive, fit negative'''
    cumulative_positive = np.cumsum(fit_positive)
    cumulative_negative = np.cumsum(fit_negative)
    return cumulative_positive/cumulative_positive[-1], cumulative_negative/cumulative_negative[-1]

from numpy.linalg import inv

def fisher(data_1, data_2, w_0 = 0):
    mu_1 = np.mean(data_1, axis=0)
    mu_2 = np.mean(data_2, axis=0)
    cov_sum = np.cov(data_1, rowvar=False) + np.cov(data_2, rowvar=False)
    w = inv(cov_sum) @ (mu_1 - mu_2)
    return w @ data_1.T + w_0, w @ data_2.T + w_0

def fisher_with_params(data_1, data_2, w_0 = 0):
    mu_1 = np.mean(data_1, axis=0)
    mu_2 = np.mean(data_2, axis=0)
    cov_sum = np.cov(data_1, rowvar=False) + np.cov(data_2, rowvar=False)
    w = inv(cov_sum) @ (mu_1 - mu_2)
    return w, w @ data_1.T + w_0, w @ data_2.T + w_0

def get_runs(residuals):
    runs = []
    run = 1
    for i in range(len(residuals)-1):
        if residuals[i] * residuals[i+1] > 0:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    return runs

def runstest(residuals):
    N_runs = len(get_runs(residuals))
    N = len(residuals)
    N_A = np.sum(residuals > 0)
    N_B = np.sum(residuals < 0)
    expected = 1 + 2*N_A*N_B/(N)
    sig_expected = (2*N_A*N_B*(2*N_A*N_B - N))/(N**2*(N-1))
    z = (N_runs - expected)/np.sqrt(sig_expected)
    p = 2*stats.norm.sf(abs(z))
    return z, p, expected, sig_expected

def plot_residuals(ax, x, res, yerrs, xlabel):
    ax_res = ax.inset_axes([0.0, -0.25, 1, 0.2])
    ax_res.errorbar(x, res, yerr = yerrs, fmt = 'o', color = 'black')
    ax_res.set(xlabel = xlabel, ylabel = 'Residuals')
    ax_res.axhline(0, color = 'red', linewidth = 2)
    return ax_res

def plot_gauss_hist(ax, bincenters, count, title, xlabel, guesses, text_loc = (0.02, 0.97)):
    binwidth = bincenters[1] - bincenters[0]
    xerr = binwidth/2
    count_err = np.sqrt(count)

    temp_chi2 = Chi2Regression(gauss, bincenters, count, count_err)
    temp_minuit = Minuit(temp_chi2, mu = guesses[0], sigma = guesses[1], N = guesses[2])
    temp_minuit.migrad()

    ax.errorbar(bincenters, count, xerr = xerr, yerr = count_err, fmt = 'o', color = 'black')
    x = np.linspace(np.min(bincenters), np.max(bincenters), 100)
    ax.plot(x, gauss(x, *temp_minuit.values[:]), color = 'red')
    ax.set(xlabel = xlabel, ylabel = f'Frequency/ {binwidth:.2f} nm', title = title)

    chi2_val, ndof, chi2_prob = evaluate_chi2(temp_minuit, len(count))
    d = {'Chi2': chi2_val,
            'Ndof': ndof,
            'Prob': chi2_prob,
            'mu': [temp_minuit.values['mu'], temp_minuit.errors['mu']],
            'sig': [temp_minuit.values['sigma'], temp_minuit.errors['sigma']],
            'N': [temp_minuit.values['N'], temp_minuit.errors['N']],
            }
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(*text_loc, text, ax, fontsize=12);

def plot_pref():
    plt.style.use('classic')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True





# Additional functions by
#Author: Christian Michelsen, NBI, 2018
#        Troels Petersen, NBI, 2019-22
def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None


from iminuit.util import make_func_code
from iminuit import describe 

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])

class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2
