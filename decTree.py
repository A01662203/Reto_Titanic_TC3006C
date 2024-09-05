def decTree(df_train):
    train = pd.read_csv('./train_clean.csv')
    test = pd.read_csv('./test_clean.csv')
    # Prepare training data
    train.drop('PassengerId', axis=1, inplace=True)
    x_train = train.drop('Survived', axis=1)
    y_train = train['Survived']

    # Set up Decision Tree model
    tree_clf = tree.DecisionTreeClassifier(max_depth=10)

    # Train model
    tree_clf.fit(x_train, y_train)

    # Export the decision tree using Graphviz
    dot_data = tree.export_graphviz(tree_clf, out_file=None, 
                                    feature_names=list(x_train.columns),  
                                    class_names=["Not Survived", "Survived"],  
                                    filled=True, rounded=True,  
                                    special_characters=True)  

    # Create a Graphviz source object
    graph = graphviz.Source(dot_data)  

    # Render and save as a PDF
    graph.render("decision_tree", format="pdf", cleanup=True)