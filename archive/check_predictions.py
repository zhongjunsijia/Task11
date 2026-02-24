from pollution_app.models import PredictionResult
from datetime import date

# Get predictions
preds = PredictionResult.objects.filter(target_date__gte=date.today()).order_by('target_date')
print(f'Found {len(preds)} predictions')

for pred in preds:
    print(f'{pred.target_date.strftime('%Y-%m-%d')}:')
    print(f'  Linear: PM2.5={pred.pm25_pred:.2f}, PM10={pred.pm10_pred:.2f}')
    print(f'  Neural Network: PM2.5={pred.pm25_nn_pred:.2f}, PM10={pred.pm10_nn_pred:.2f}')
    print(f'  Random Forest: PM2.5={pred.pm25_rf_pred:.2f}, PM10={pred.pm10_rf_pred:.2f}')
    print()