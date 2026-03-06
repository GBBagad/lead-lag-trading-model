# backend/strategy/validation.py

def validate_lag(optimal_lag, corrs, pvals):

    corr_value = corrs[optimal_lag]
    p_value = pvals[optimal_lag]

    if abs(corr_value) > 0.03 and p_value < 0.5:
        return True
    else:
        return False