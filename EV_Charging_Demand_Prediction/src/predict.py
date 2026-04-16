from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def make_predictions(model, X_test, y_test):
    """
    Generate predictions using data

    parameters:
        model,X_test,y_test

    predict:
        mae, rmse, r2, mape
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    non_zero_mask=y_test != 0
    if non_zero_mask.sum()>0:
        mape=np.mean(
            np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])
        )*100
    else:
        mape=float('nan')

    metrics={
        'mae':round(mae,4),
        'rmse':round(rmse,4),
        'r2':round(r2,4),
        'mape':round(mape,2),
    }

    return predictions, metrics


def get_feature_importance(model, feature_names):
    """
    extract feature importance from the trained model.
    {feature_name:mportance_score}
    """
    if hasattr(model,'feature_importances_'):
        importances=model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances=np.abs(model.coef_)
        total=importances.sum()
        if total>0:
            importances=importances/total
    else:
        return {}

    return {feature_names[i]:importances[i] for i in range(len(feature_names))}