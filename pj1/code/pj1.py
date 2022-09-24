import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, learning_curve,validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np


data_wine=pd.read_csv("data/winequality-white.csv")
data_default=pd.read_csv("data/Italy_default_prior_2004.csv")

#for shorter code, here I split with dataset 1 and dataset 2, the processing should be similar
dataset_num=1 #0 for wine quality, 1 for italian default
algo_num=5 #1 for Decision Tree, 2 for Neural Network, 3 for boosting, 4 for SNM, 5 for kNN
if dataset_num==0: #wine quality set
    data=data_wine.sample(frac=1).reset_index(drop=True)#shuffle data
    data_x=data.loc[:,data.columns!="quality"]
    data_y=data.loc[:,data.columns=="quality"]
else:
    data=data_default.sample(frac=1).reset_index(drop=True)#shuffle data
    data_x=data.loc[:,data.columns!="DefaultIndex"]
    le = LabelEncoder()
    for column in data_x.columns:
        data_x[column] = le.fit_transform(data_x[column])
        mapping = dict(zip(le.classes_, range(1, len(le.classes_) + 1)))
        # print(mapping)
    data_y = data.loc[:, data.columns == "DefaultIndex"]
#split train/test data
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.4,train_size=0.6,random_state=0).copy()
##############################################################################
#Decision Tree:
if algo_num==1:
    output_list=[]
    for i in [1,2,3,4,5,6,8,10,20,50]:
        start = datetime.now()
        params = {'criterion': ['gini', 'entropy']}
        clf = tree.DecisionTreeClassifier(max_depth=i, class_weight='balanced', splitter='best', min_samples_leaf=1)
        cv_clf = GridSearchCV(clf, param_grid=params, refit=True, cv=10)

        cv_clf.fit(x_train, y_train)
        train_score = cv_clf.score(x_train, y_train)
        test_score = cv_clf.score(x_test, y_test)
        y_predict = cv_clf.predict(x_test)
        end = datetime.now()
        run_time = (end - start).total_seconds()
        confusion_results = confusion_matrix(y_test, y_predict)
        print(confusion_results)

        cv_estimator = cv_clf.best_estimator_

        cv_curve = learning_curve(cv_clf.best_estimator_, x_train, y_train, cv=10, train_sizes=np.linspace(0.1, 1.0, 10))

        train_sizes, train_scores, test_scores = cv_curve
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.grid()
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.ylim(0,1)
        plt.title("Traing score, max_depth="+str(i))

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="c",label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="y",label="Average Test Score")

        tree_size = cv_clf.best_estimator_.tree_.node_count
        output_list.append([i,cv_clf.best_params_['criterion'], tree_size,train_score, test_score, run_time])
        if dataset_num==0:
            plt.savefig("result/decision_tree/wine_"+str(i)+".png")
        else:
            plt.savefig("result/decision_tree/default_" + str(i) + ".png")
    output_df=pd.DataFrame(output_list,columns=["Max Depth","Best Split Method","Tree Size","Training Score","Test Score","Running Time"])
    if dataset_num==0:
        output_df.to_csv("result/decision_tree/wine.csv")
    else:
        output_df.to_csv("result/decision_tree/default.csv")
######end of decision tree###################

######start of neural network##############
if algo_num==2:
    #experiment without tuning
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)
    clf_nn.fit(x_train, y_train)
    y_predict = clf_nn.predict(x_test)
    nn_accuracy = accuracy_score(y_test, y_predict)
    print('Accuracy without tuning is %.2f%%' % (nn_accuracy * 100))
    ########
    alpha_range = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
    train_scores, test_scores = validation_curve(clf_nn, x_train, y_train, cv=5,param_name="alpha", param_range=alpha_range)

    plt.figure()
    plt.semilogx(alpha_range, np.mean(train_scores, axis=1), label='Training score') #get this method about plotting from https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.semilogx.html
    plt.semilogx(alpha_range, np.mean(test_scores, axis=1), label='Testing score')
    plt.title('Alpha vs Traing Score (anchoring learning rate)')
    plt.xlabel('Alpha')
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.grid()
    plt.legend(loc="best")
    if dataset_num==0:
        plt.savefig("result/neural_network/anchoring_lr_wine.png")
    if dataset_num==1:
        plt.savefig("result/neural_network/anchoring_lr_default.png")

    lr_range = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1]
    train_scores, test_scores = validation_curve(clf_nn, x_train, y_train, cv=5,param_name="learning_rate_init",param_range=lr_range)
    plt.figure()
    plt.semilogx(lr_range, np.mean(train_scores, axis=1),label='Training score')
    plt.semilogx(lr_range, np.mean(test_scores, axis=1), label='Testing score')
    plt.title('Learning Rate vs Traing Score (anchoring alpha)')
    plt.xlabel('Learning Rate')
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(loc="best")
    if dataset_num == 0:
        plt.savefig("result/neural_network/anchoring_alpha_wine.png")
    if dataset_num == 1:
        plt.savefig("result/neural_network/anchoring_alpha_default.png")

    ##parameter tuning
    if dataset_num==0:
        alpha_range =  [0.01,0.1,1,2,5,10]
        # alpha_range = [0.01]
        lr_range = [0.0001,0.001,0.01,0.1,1]
        # lr_range = [0.01]
    if dataset_num==1:
        alpha_range = [0.001,0.005,0.01,0.1]
        # alpha_range = [0.001]
        lr_range = [0.000001,0.00001,0.0001]
        # lr_range = [0.00001]
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)
    tuning_parmas = {'alpha': alpha_range, 'learning_rate_init': lr_range}
    clf_nn = GridSearchCV(clf_nn, param_grid=tuning_parmas, cv=5)
    start_time= datetime.now()
    clf_nn.fit(x_train, y_train)
    end_time = datetime.now()
    run_time=(end_time-start_time).total_seconds()
    print('Completed training in %f seconds' % run_time)
    best_params = clf_nn.best_params_
    best_clf_nn = clf_nn
    print("Best parameters set found on development set:")
    print(best_params)

    cv_curve = learning_curve(best_clf_nn, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5))

    train_sizes, train_scores, test_scores = cv_curve
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.grid()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Traing score for Neural Network " + str(best_params))

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="c", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="y", label="Average Test Score")
    if dataset_num == 0:
        plt.savefig("result/neural_network/learning_curve_wine.png")
    if dataset_num == 1:
        plt.savefig("result/neural_network/learning_curve_default.png")
    #####end of param tuning####

    ##loss curve#####
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1, warm_start=True)
    #clf_nn.set_params(alpha=best_params['alpha'], learning_rate_init=best_params['learning_rate_init'])
    if dataset_num==1:
        clf_nn.set_params(alpha=[0.001], learning_rate_init=[0.00001])
    if dataset_num==0:
        clf_nn.set_params(alpha=[0.01], learning_rate_init=[0.01])
    num_iter = 500
    train_loss,train_scores,val_scores=[],[],[]
    ##generate validation dataset:
    x_train_sub, x_validate, y_train_sub, y_validate = train_test_split(x_train, y_train, test_size=0.4, train_size=0.6,random_state=0).copy()
    for i in range(num_iter):
        clf_nn.fit(x_train_sub, y_train_sub)
        train_loss.append(clf_nn.loss_)
        train_scores.append(accuracy_score(y_train_sub,clf_nn.predict(x_train_sub)))
        val_scores.append(accuracy_score(y_validate, clf_nn.predict(x_validate)))
    y_pred = clf_nn.predict(x_test)
    nn_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of neural network on test dataset is  %.2f%%' % (nn_accuracy * 100))
    plt.figure()
    plt.plot(np.arange(num_iter) + 1, train_loss)
    plt.title('Loss curve for neural network')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.grid()
    if dataset_num == 0:
        plt.savefig("result/neural_network/loss_curve_wine.png")
    if dataset_num == 1:
        plt.savefig("result/neural_network/loss_curve_default.png")
    plt.figure()
    plt.plot(np.arange(num_iter) + 1, train_scores, label='Training score')
    plt.plot(np.arange(num_iter) + 1, val_scores, label='Validation score')
    plt.title('Training and validation score curve for neural network')
    plt.xlabel('Epochs')
    plt.ylabel("Training score")
    plt.grid()
    plt.legend(loc="best")
    if dataset_num == 0:
        plt.savefig("result/neural_network/predict_score_curve_wine.png")
    if dataset_num == 1:
        plt.savefig("result/neural_network/predict_score_curve_default.png")
    ##########

##############end of neural network###########################

#########Start of boosting##############
#as boosting is similar to decision tree, most of code in this chunk is copied from previous chunk.
if algo_num==3:
    output_list=[]
    estimators = [1, 5, 10, 20, 50, 100]
    learning_rate = [0.001,0.01,0.1, 1, 10]
    for i in [1,5,8,10,20,50]:
        start = datetime.now()
        params = {'n_estimators': estimators,'learning_rate': learning_rate}
        if dataset_num==0:
            if i <=10:
                clf_no_boost = tree.DecisionTreeClassifier(max_depth=i, criterion='entropy', splitter='best')
            else:
                clf_no_boost = tree.DecisionTreeClassifier(max_depth=i, criterion='gini', splitter='best')
        if dataset_num==1:
            if i <=10:
                clf_no_boost = tree.DecisionTreeClassifier(max_depth=i, criterion='gini', splitter='best')
            else:
                clf_no_boost = tree.DecisionTreeClassifier(max_depth=i, criterion='entropy', splitter='best')
        clf = AdaBoostClassifier(base_estimator=clf_no_boost)
        cv_clf = GridSearchCV(clf, param_grid=params, refit=True, cv=5)

        cv_clf.fit(x_train, y_train)
        train_score = cv_clf.score(x_train, y_train)
        test_score = cv_clf.score(x_test, y_test)
        y_predict = cv_clf.predict(x_test)
        end = datetime.now()
        run_time = (end - start).total_seconds()
        confusion_results = confusion_matrix(y_test, y_predict)
        print(confusion_results)

        cv_estimator = cv_clf.best_estimator_

        cv_curve = learning_curve(cv_clf.best_estimator_, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

        train_sizes, train_scores, test_scores = cv_curve
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.grid()
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.ylim(0,1)
        plt.title("Traing score, max_depth="+str(i))

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="c",label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="y",label="Average Test Score")
        plt.legend(loc="best")

        tree_size = len(cv_clf.best_estimator_)
        output_list.append([i,cv_clf.best_params_['n_estimators'], cv_clf.best_params_['learning_rate'],tree_size,train_score, test_score, run_time])
        if dataset_num==0:
            plt.savefig("result/boosting/wine_"+str(i)+".png")
        else:
            plt.savefig("result/boosting/default_" + str(i) + ".png")
    output_df=pd.DataFrame(output_list,columns=["Max Depth","Number of estimator","Learning rate"," Size","Training Score","Test Score","Running Time"])
    if dataset_num==0:
        output_df.to_csv("result/boosting/wine.csv")
    else:
        output_df.to_csv("result/boosting/default.csv")
#############end of boosting#####################

############start of SVM##########################
if algo_num==4:
    output_list = []
    #tuning kernels##
    kernels = ['poly',"linear",'rbf','sigmoid']
    gamma = "auto"
    for kernel in kernels:
        print(kernel)
        start = datetime.now()
        clf = svm.SVC(kernel=kernel, gamma=gamma)
        cv_clf = GridSearchCV(clf, param_grid={},cv=10)

        cv_clf.fit(x_train, y_train)
        train_score = cv_clf.score(x_train, y_train)
        test_score = cv_clf.score(x_test, y_test)
        y_predict = cv_clf.predict(x_test)
        end = datetime.now()
        run_time = (end - start).total_seconds()
        confusion_results = confusion_matrix(y_test, y_predict)
        print(confusion_results)

        cv_estimator = cv_clf.best_estimator_

        cv_curve = learning_curve(cv_clf.best_estimator_, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

        train_sizes, train_scores, test_scores = cv_curve
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.grid()
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("Traing score, kernel=" + str(kernel))

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="c", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="y", label="Average Test Score")
        plt.legend(loc="best")

        output_list.append([kernel,  train_score,test_score, run_time])
        if dataset_num == 0:
            plt.savefig("result/SVM/wine_"  + str(kernel) + ".png")
        else:
            plt.savefig("result/SVM/default_" + str(kernel) + ".png")
    output_df = pd.DataFrame(output_list,columns=["Kernel", "Training Score","Test Score", "Running Time"])
    if dataset_num == 0:
        output_df.to_csv("result/SVM/wine.csv")
    else:
        output_df.to_csv("result/SVM/default.csv")
    ###tuning gammas with kernel anchoring as 'rbf'
    output_list = []
    gammas = [0.01,0.05,0.1,0.5,1,2]
    for gamma in gammas:
        print(gamma)
        start = datetime.now()
        clf = svm.SVC(kernel='rbf', gamma=gamma)
        cv_clf = GridSearchCV(clf, param_grid={}, cv=5)

        cv_clf.fit(x_train, y_train)
        train_score = cv_clf.score(x_train, y_train)
        test_score = cv_clf.score(x_test, y_test)
        y_predict = cv_clf.predict(x_test)
        end = datetime.now()
        run_time = (end - start).total_seconds()
        confusion_results = confusion_matrix(y_test, y_predict)
        print(confusion_results)

        cv_estimator = cv_clf.best_estimator_

        cv_curve = learning_curve(cv_clf.best_estimator_, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

        train_sizes, train_scores, test_scores = cv_curve
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.grid()
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("Traing score, gamma=" + str(gamma))

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="c", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="y", label="Average Test Score")
        plt.legend(loc="best")

        output_list.append([gamma, train_score, test_score, run_time])
        if dataset_num == 0:
            plt.savefig("result/SVM/wine_gamma" + str(gamma) + ".png")
        else:
            plt.savefig("result/SVM/default_gamma" + str(gamma) + ".png")
    output_df = pd.DataFrame(output_list, columns=["Kernel", "Training Score", "Test Score", "Running Time"])
    if dataset_num == 0:
        output_df.to_csv("result/SVM/wine_gamma.csv")
    else:
        output_df.to_csv("result/SVM/default_gamma.csv")
    ########end of SVM##############

###########start of kNN############
if algo_num==5:
    output_list = []
    weights = ["uniform","distance"]
    ks=[1,2,3,4,5,6,7,8,9,10,20,50,100]
    for weight in weights:
        for k in ks:
            print(weight+str(k))
            start = datetime.now()
            clf = KNeighborsClassifier(k, weights=weight)
            cv_clf = GridSearchCV(clf, param_grid={}, cv=10)

            cv_clf.fit(x_train, y_train)
            train_score = cv_clf.score(x_train, y_train)
            test_score = cv_clf.score(x_test, y_test)
            y_predict = cv_clf.predict(x_test)
            end = datetime.now()
            run_time = (end - start).total_seconds()
            confusion_results = confusion_matrix(y_test, y_predict)
            print(confusion_results)

            cv_estimator = cv_clf.best_estimator_

            cv_curve = learning_curve(cv_clf.best_estimator_, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

            train_sizes, train_scores, test_scores = cv_curve
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)

            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.figure()
            plt.grid()
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.ylim(0, 1)
            plt.title("Traing score, weight=" + weight +" k= "+str(k))

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                             alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")

            plt.plot(train_sizes, train_scores_mean, 'o-', color="c", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="y", label="Average Test Score")
            plt.legend(loc="best")

            output_list.append([weight,k, train_score, test_score, run_time])
            if dataset_num == 0:
                plt.savefig("result/knn/wine_" + weight+" k_ " +str(k) + ".png")
            else:
                plt.savefig("result/knn/default_" + weight+" k_ " +str(k)  + ".png")
    output_df = pd.DataFrame(output_list, columns=["weight","k", "Training Score", "Test Score", "Running Time"])
    if dataset_num == 0:
        output_df.to_csv("result/knn/wine.csv")
    else:
        output_df.to_csv("result/knn/default.csv")

