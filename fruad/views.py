from django.shortcuts import render
import json
import joblib
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

@csrf_exempt
def predict(request):
    if request.POST is not None:

        model = joblib.load('model.sav')
        scaler = joblib.load('scaler.sav')

        var = json.loads(request.body)
        time = float(var['t'])
        amount = float(var['amount'])
        time1 = scaler.fit_transform(np.array([time]).reshape(-1,1))
        amount1 = scaler.fit_transform(np.array([amount]).reshape(-1,1))
        v1 = float(var['v1'])
        v2 = float(var['v2'])
        v3 = float(var['v3'])
        v4 = float(var['v4'])
        v5 = float(var['v5'])
        v6 = float(var['v6'])
        v7 = float(var['v7'])
        v8 = float(var['v8'])
        v9 = float(var['v9'])
        v10 = float(var['v10'])
        v11 = float(var['v11'])
        v12 = float(var['v12'])
        v13 = float(var['v13'])
        v14 = float(var['v14'])
        v15 = float(var['v15'])
        v16 = float(var['v15'])
        v17 = float(var['v17'])
        v18 = float(var['v18'])
        v19 = float(var['v19'])
        v20 = float(var['v20'])
        v21 = float(var['v21'])
        v22 = float(var['v22'])
        v23 = float(var['v23'])
        v24 = float(var['v24'])
        v25 = float(var['v25'])
        v26 = float(var['v26'])
        v27 = float(var['v27'])
        v28 = float(var['v28'])

        pred = model.predict(np.array([[time1, amount1, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                                       v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]]))


        print("pred is",pred[0])
        print(type(pred[0]))

        if pred[0] == 0:
            result = "Not a Fruad"
        else:
            result = "Fruad"

        return JsonResponse({"pred" : result}, safe = False)
