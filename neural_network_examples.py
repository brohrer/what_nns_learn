import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np

import activation_functions as act


class Layer(object):
    def __init__(
        self,
        activation_function="tanh",
        debug=False,
        m_inputs=1,
        n_outputs=1,
    ):
        self.debug = debug
        self.m_inputs = m_inputs
        self.n_outputs = n_outputs
        # Make a best guess at the activation function based on the first
        # letter.
        first_letter = activation_function.lower()[0]
        if first_letter in ["l", "s"]:
            # Catch "logit", "logistic", and "sigmoid"
            # A bit of pedantry: "sigmoid" literally means "s-shaped"
            # and describes both logistic and hyperbolic tangent
            # functions.
            self.activation_function = act.logit
            act_fun_name = "sigmoid"
        elif first_letter in ["t", "h"]:
            # Catch "tanh" and "hyperbolic tangent"
            self.activation_function = act.tanh
            act_fun_name = "hyperbolic tangent"
        elif first_letter in ["n", "p", "i"]:
            # Catch "none", "passthrough", and "identity"
            self.activation_function = act.none
            act_fun_name = "linear"
        else:
            self.activation_function = act.relu
            act_fun_name = "rectified linear unit"
        if self.debug:
            print("Using", act_fun_name, "activation function")

        # Choose random weights between -1 and 1.
        # Inputs match to rows. Outputs match to columns.
        # Add one to m_inputs to account for the bias term.
        self.weights = np.random.sample(
            size=(self.m_inputs + 1, self.n_outputs)) * 2 - 1
        # Make the weights array 3D so that it can handle entire
        # batches at once.
        self.weights = self.weights[:, :, np.newaxis]

    def forward_prop(self, inputs):
        """
        Propagate the inputs forward through the network.

        Parameters
        inputs: ndarray
            Expects a 2D array. Rows are inputs,
            columns are separate sets of inputs, each a distinct example.
            The entire array represents one batch of inputs.
            If inputs is one-dimensional, it's assumed to be a batch of one.

        Returns
        y: ndarray
            Returns a 2D array of outputs. Rows are individual outputs,
            columns are distinct examples.
        """
        # Make a copy so as not to modify the original values.
        inputs = inputs.copy()
        if len(inputs.shape) == 1:
            inputs = inputs[:, np.newaxis]
        m_inputs, p_examples = inputs.shape
        assert m_inputs == self.m_inputs

        bias = np.ones((1, p_examples))
        x = np.concatenate((inputs, bias), axis=0)

        x = x[:, np.newaxis, :]
        u = x * self.weights
        v = np.sum(u, axis=0)
        y = self.activation_function(v)

        return y


class FCNN(object):
    def __init__(self, n_nodes, activation_function):
        self.n_nodes = n_nodes
        self.n_layers = len(n_nodes) - 1

        self.layers = []
        for l in range(self.n_layers):
            self.layers.append(
                Layer(
                    m_inputs=n_nodes[l],
                    n_outputs=n_nodes[l + 1],
                    activation_function=activation_function,
                ))

    def forward_prop(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward_prop(y)
        return y


def make_2D_functions(
    activation_function="tanh",
    dpi=150,
    n_plots=11,
    n_nodes=[1, 1],
    n_x=99,
    title="NN 2D functions",
):
    assert n_nodes[0] == 1
    dirname = "_".join(title.lower().split()) + "_plots"
    try:
        os.mkdir(dirname)
    except Exception:
        pass

    plt.style.use('dark_background')
    plt.figure(3287, figsize=(8, 4.5))

    xmin, xmax, ymin, ymax = (-6, 6, -1.2, 1.2)
    x_1D = np.linspace(xmin, xmax, 100)[np.newaxis, :]

    for i_plot in range(n_plots):
        network = FCNN(n_nodes, activation_function=activation_function)
        y = network.forward_prop(x_1D)

        plt.clf()
        plt.plot([0, 0], [ymin, ymax], linewidth=.5)
        plt.plot([xmin, xmax], [0, 0], linewidth=.5)
        for y_j in range(y.shape[0]):
            plt.plot(x_1D[0, :], y[y_j, :], linewidth=3)

        plt.xlabel('x (input)')
        plt.xlim(xmin, xmax)
        plt.ylabel('y (output)')
        plt.ylim(ymin, ymax)
        plt.title(title)

        filename = os.path.join(
            dirname,
            "{0}_{1}.png".format(dirname, i_plot))
        plt.savefig(filename, dpi=dpi)


def make_3D_functions(
    activation_function="tanh",
    dpi=150,
    n_plots=11,
    n_nodes=[2, 1],
    n_x1=99,
    n_x2=101,
    title="NN 3D functions",
):
    assert n_nodes[0] == 2
    dirname = "_".join(title.lower().split()) + "_plots"
    try:
        os.mkdir(dirname)
    except Exception:
        pass

    plt.style.use('dark_background')
    fig = plt.figure(3287, figsize=(8, 4.5))

    xmin, xmax, ymin, ymax = (-6, 6, -2, 1)
    x1, x2 = np.meshgrid(np.linspace(xmin, xmax, n_x1),
                         np.linspace(xmin, xmax, n_x2))
    x_2D = np.concatenate((x1.ravel()[np.newaxis, :],
                           x2.ravel()[np.newaxis, :]), axis=0)

    for i_plot in range(n_plots):
        network = FCNN(n_nodes, activation_function=activation_function)
        y = network.forward_prop(x_2D)
        y_2D = np.reshape(y, (n_x2, n_x1))

        plt.clf()
        ax = fig.gca(projection='3d')
        ax.view_init(20, -55)
        ax.plot_surface(
            x1, x2, y_2D,
            cmap=cm.RdYlGn,
            ccount=300,
            rcount=300,
            vmin=-1,
            vmax=1,
        )
        ax.contour(
            x1, x2, y_2D,
            zdir='z',
            offset=ymin,
            cmap=cm.RdYlGn,
            vmin=-1,
            vmax=1,
        )

        ax.set_xlabel('x_1 (input 1)')
        ax.set_xlim(xmin, xmax)
        ax.set_ylabel('x_2 (input 2)')
        ax.set_ylim(xmin, xmax)
        ax.set_zlabel('y (output)')
        ax.set_zlim(ymin, ymax)
        plt.title(title)

        filename = os.path.join(
            dirname,
            "{0}_{1}.png".format(dirname, i_plot))
        plt.savefig(filename, dpi=dpi)


n_plots = 49

# 0. Linear regression
# (with classification) perceptron
make_2D_functions(
    activation_function="identity",
    n_nodes=[1, 1],
    n_plots=n_plots,
    title="Linear 1-layer 1-input 1-output",
)
# 1. m-dimensional linear regression
# (with classification) m-dimensional perceptron
make_3D_functions(
    activation_function="identity",
    n_nodes=[2, 1],
    n_plots=n_plots,
    title="Linear 1-layer 2-input 1-output",
)
# (with classification) multi-layer perceptron
make_2D_functions(
    activation_function="identity",
    n_nodes=[1, 64, 1],
    n_plots=n_plots,
    title="Linear 2-layer 1-input 1-output",
)
make_2D_functions(
    activation_function="identity",
    n_nodes=[1, 16, 16, 1],
    n_plots=n_plots,
    title="Linear 3-layer 1-input 1-output",
)
# 3. 0 with logistic activation function
# (with classification) logistic regression, like perceptron
make_2D_functions(
    activation_function="logistic",
    n_nodes=[1, 1],
    n_plots=n_plots,
    title="Logistic 1-layer 1-input 1-output",
)
# 4. (with classification) m-dimensional logistic regression
# like m-dimensional perceptron
make_3D_functions(
    activation_function="logistic",
    n_nodes=[2, 1],
    n_plots=n_plots,
    title="Logistic 1-layer 2-input 1-output",
)
# 5. 0 with hyperbolic tangent activation function
# (with classification) like logistic regression and perceptron
make_2D_functions(
    activation_function="tanh",
    n_nodes=[1, 1],
    n_plots=n_plots,
    title="Hyperbolic tangent 1-layer 1-input 1-output",
)
# 6. (with classification) multi-layer perceptron (MLP)
make_2D_functions(
    activation_function="tanh",
    n_nodes=[1, 64, 1],
    n_plots=n_plots,
    title="Hyperbolic tangent 2-layer 1-input 1-output",
)
make_2D_functions(
    activation_function="tanh",
    n_nodes=[1, 16, 16, 1],
    n_plots=n_plots,
    title="Hyperbolic tangent 3-layer 1-input 1-output",
)
# 7. (with classification) m-dimensional MLP
make_3D_functions(
    activation_function="tanh",
    n_nodes=[2, 64, 1],
    n_plots=n_plots,
    title="Hyperbolic tangent 2-layer 2-input 1-output",
)
make_3D_functions(
    activation_function="tanh",
    n_nodes=[2, 16, 16, 1],
    n_plots=n_plots,
    title="Hyperbolic tangent 3-layer 2-input 1-output",
)
# 7.5 (with classification) multi-class classifier
make_2D_functions(
    activation_function="tanh",
    n_nodes=[1, 16, 16, 2],
    n_plots=n_plots,
    title="Hyperbolic tangent 3-layer 1-input 2-output",
)
make_2D_functions(
    activation_function="tanh",
    n_nodes=[1, 16, 16, 3],
    n_plots=n_plots,
    title="Hyperbolic tangent 3-layer 1-input 3-output",
)
# 8. fully connected layer
make_3D_functions(
    activation_function="tanh",
    n_nodes=[2, 16, 16, 1],
    n_plots=n_plots,
    title="Hyperbolic tangent 3-layer 2-input 1-output",
)
# 9. fully connected layer with ReLU
make_3D_functions(
    activation_function="relu",
    n_nodes=[2, 16, 16, 1],
    n_plots=n_plots,
    title="Rectified linear units 3-layer 2-input 1-output",
)
