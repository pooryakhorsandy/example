from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import pickle
import 

def hello(request):
    if request.method == 'POST':
        # If the form is submitted, perform the calculation
        LC = float(request.POST.get('LC', 0))
        PC = float(request.POST.get('PC', 0))
        LD = float(request.POST.get('L/D', 0))
        BT = float(request.POST.get('B/T', 0))
        LB = float(request.POST.get('L/B', 0))
        Fr = float(request.POST.get('Fr', 0))
        with open(r'xgb_model.pkl', 'rb') as model_file:
            xgb_model = pickle.load(model_file)
        features=[LC,PC,LD,BT,LB,Fr]
        input_data = np.array(features).reshape(1, -1)
        result = xgb_model.predict(input_data)[0]

        return render(request, 'index.html', {'result': result})
    else:
        # If it's a GET request, just render the template
        return render(request, 'index.html')
