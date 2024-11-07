from sklearn.metrics import classification_report

def metrics(y_train, pred_train , y_val, pred_val):
    print('______________________________________________________________________')
    print('                                TRAIN                                 ')
    print('----------------------------------------------------------------------')
    print(classification_report(y_train, pred_train))



    print('______________________________________________________________________')
    print('                                VALIDATION                                 ')
    print('----------------------------------------------------------------------')
    print(classification_report(y_val, pred_val))