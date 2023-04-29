
import numpy as np
from simulation import MyData
from typing import Tuple

#|%%--%%| <5dHwIIddgG|Wk9RkiI9s7>

class OrdinaryLinearSquares:
    def __init__(self, training_data: MyData):
        self.training_data = training_data
        self.estimated_coefficients = self.fit_ols()
        self.y_prediction = self.get_predicted_labels()
    def fit_ols(self) -> np.ndarray:
        x_features = self.training_data.x_features
        y_labels = self.training_data.y_labels
        estimated_coefficients = np.linalg.inv(x_features.T @ x_features) @ \
                                 x_features.T @ y_labels
        return estimated_coefficients
    def get_predicted_labels(self, x=None) -> np.ndarray:
        if x is None:
            x = self.training_data.x_features
        y_prediction = x @ self.estimated_coefficients
        return y_prediction
    def get_prediction_with_residuals(
            self, test_data: MyData
            ) -> Tuple[np.ndarray, np.ndarray]:
        # y - \hat y
        x = test_data.x_features
        y_true = test_data.y_labels
        y_prediction = self.get_predicted_labels(x)
        diff = y_true - y_prediction 
        return y_prediction, diff
    def get_full_conformal_set(self, alpha):
        X = self.training_data.x_features[:-1]
        x = self.training_data.x_features[-1]
        Y = self.training_data.y_labels[:-1]

        XtX = X.T.dot(X) + np.outer(x,x)
        a = Y - X.dot(np.linalg.solve(XtX,X.T.dot(Y)))
        b = -X.dot(np.linalg.solve(XtX,x))
        a1 = -x.T.dot(np.linalg.solve(XtX,X.T.dot(Y)))
        b1 = 1 - x.T.dot(np.linalg.solve(XtX,x))
        # if we run weighted least squares on (X[1,],Y[1]),...(X[n,],Y[n]),(x,y)
        # then a + b*y = residuals of data points 1,..,n
        # and a1 + b1*y = residual of data point n+1

        # Sort all the nodes
        y_knots = np.sort(np.unique(np.r_[((a-a1)/(b1-b))[b1-b!=0],((-a-a1)/(b1+b))[b1+b!=0]]))
        # Check at each node whether the residual of the test (X,y_knot) is below 1-alpha quantile
        n = len(self.training_data.y_labels) - 1
        tmp = (np.abs(np.outer(a1+b1*y_knots,np.ones(n))) >
               np.abs(np.outer(np.ones(len(y_knots)),a)+np.outer(y_knots,b))) / (n+1)
        # y_inds_keep = np.where(tmp.sum(1) <= 1-alpha )[0]
        y_inds_keep = np.where(tmp.sum(1) <= alpha )[0]
        # Form the interval with all "Yes" nodes as the prediction interval
        # print(y_inds_keep.min(), y_inds_keep.max())
        y_PI = np.array([y_knots[y_inds_keep.min()],y_knots[y_inds_keep.max()]])
        set_length = y_PI[1] - y_PI[0]
        # print(y_inds_keep)
        return y_PI, set_length



#|%%--%%| <Wk9RkiI9s7|wlzVKsKxO7>


# """Test"""

# from simulation import MyData
# training_data = sample_dataset(7, 7)
# model_ols = OrdinaryLinearSquares(training_data)
# (model_ols.y_prediction - training_data.y_labels).mean()

