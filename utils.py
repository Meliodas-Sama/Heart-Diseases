
def Evaluate(classifier, df):
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import classification_report
    import GaussianNB
    from sklearn.model_selection import train_test_split

    # st.set_option('deprecation.showPyplotGlobalUse', False)

    
    data = df.copy()

    features = data.loc[:,'age':'exercice_angina']
    target = data.loc[:, 'disease']

    X_unprocessed = features.values
    y_unprocessed = target.values

    X_train, X_test, y_train, y_test = train_test_split(X_unprocessed, y_unprocessed, random_state=1, stratify=y_unprocessed)

    y_predicted, y_probability_estimates = GaussianNB.Predict(classifier,X_test)

    accuracy = accuracy_score(y_test,y_predicted)
    classification_rep = classification_report(y_test,y_predicted)
    con_matrix = confusion_matrix(y_test,y_predicted)

    y_ones = [0 if (x=='negative') else 1 for x in y_test]
    roc_auc = roc_auc_score(y_ones,y_probability_estimates[:,0])
    fpr, tpr, _ = roc_curve(y_ones,y_probability_estimates[:,0])

    # ploting confustion matrix
    fig = plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    sns.heatmap(con_matrix, annot=True, fmt="d" )
    plt.ylabel("Real value")
    plt.xlabel("Predicted value")
    plt.show()
    st.pyplot(fig)

    # print scores
    st.write ("accuracy  score: {} %".format(accuracy))
    st.write ("auc  score: {} ".format(roc_auc))
    st.write(classification_rep)

    # print ROC curve
    st.write("\n\n\n\n")
    st.markdown("### ROC curve")
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange',lw=2, 
            label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    st.pyplot(fig)
