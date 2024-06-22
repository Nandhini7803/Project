from flask import Flask, render_template, flash, request, session, send_file
from flask import render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
import mysql.connector
import sys
from ecies.utils import generate_key
from ecies import encrypt, decrypt
import base64, os
import pickle
import numpy as np

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Home")
def Home():
    return render_template('index.html')


@app.route("/OwnerLogin")
def OwnerLogin():
    return render_template('OwnerLogin.html')


@app.route("/NewOwner")
def NewOwner():
    return render_template('NewOwner.html')


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route("/UploadDataSet")
def UploadDataSet():
    return render_template('UploadDataSet.html')


@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')


@app.route("/NewUser")
def NewUser():
    return render_template('NewUser.html')


@app.route("/Cancer")
def Cancer():
    return render_template('Cancer.html')


@app.route("/Diabetes")
def Diabetes():
    return render_template('Diabetes.html')


@app.route("/Heart")
def Heart():
    return render_template('Heart.html')


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
        if request.form['uname'] == 'admin' and request.form['password'] == 'admin':

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb ")
            data = cur.fetchall()
            return render_template('AdminHome.html', data=data)

        else:

            alert = 'Username or Password is wrong'
            return render_template('goback.html', data=alert)


@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb ")
    data = cur.fetchall()
    return render_template('AdminHome.html', data=data)


@app.route("/OwnerInfo")
def OwnerInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM ownertb ")
    data = cur.fetchall()
    return render_template('OwnerInfo.html', data=data)


@app.route("/UploadDataset")
def UploadDataset():
    return render_template('ViewExcel.html')


@app.route("/excelpost", methods=['GET', 'POST'])
def excelpost():
    if request.method == 'POST':
        typ = request.form['typ']

        if typ == "Breast":
            file = request.files['fileupload']
            file_extension = file.filename.split('.')[1]
            print(file_extension)
            # file.save("static/upload/" + secure_filename(file.filename))

            import pandas as pd
            import matplotlib.pyplot as plt
            df = ''
            if file_extension == 'xlsx':
                df = pd.read_excel(file.read(), engine='openpyxl')
            elif file_extension == 'xls':
                df = pd.read_excel(file.read())
            elif file_extension == 'csv':
                df = pd.read_csv(file)

            # import pandas as pd
            import matplotlib.pyplot as plt

            # read-in data
            # data = pd.read_csv('./test.csv', sep='\t') #adjust sep to your needs

            # df = pd.read_csv('./Datasetsbreast.csv')

            def clean_dataset(df):
                assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
                df.dropna(inplace=True)
                indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
                return df[indices_to_keep].astype(np.float64)

            df = clean_dataset(df)

            # import pandas as pd
            import matplotlib.pyplot as plt

            # read-in data
            # data = pd.read_csv('./test.csv', sep='\t') #adjust sep to your needs

            import seaborn as sns
            sns.countplot(df['Class'], label="Count")
            plt.show()

            # count occurences
            # occurrences = df.loc[:, 'Outcome'].value_counts()

            # plot histogram
            # plt.bar(occurrences.keys(), occurrences)
            # plt.show()

            # Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
            df_copy = df.copy(deep=True)
            df_copy[['Clumpthickness', 'Uniformityofcellsize', 'Uniformityofcellshape', 'MarginalAdhesion',
                     'SingleEpithelialcellsize', 'BareNuclei', 'Blandchromatin', 'NormalNucleoli', 'Mitoses']] = \
                df_copy[
                    ['Clumpthickness', 'Uniformityofcellsize', 'Uniformityofcellshape', 'MarginalAdhesion',
                     'SingleEpithelialcellsize', 'BareNuclei', 'Blandchromatin', 'NormalNucleoli', 'Mitoses']].replace(
                    0,
                    np.NaN)

            # Model Building
            from sklearn.model_selection import train_test_split
            df.drop(df.columns[np.isnan(df).any()], axis=1)
            X = df.drop(columns='Class')
            y = df['Class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

            from sklearn.svm import SVC
            classifier = SVC()
            from sklearn.metrics import classification_report
            # classifier = RandomForestClassifier(n_estimators=20)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            print(classification_report(y_test, y_pred))
            # Creating a pickle file for the classifier
            filename = 'breast-prediction-rfc-model.pkl'
            pickle.dump(classifier, open(filename, 'wb'))

            Prescription = ''

            filename = 'breast-prediction-rfc-model.pkl'
            classifier = pickle.load(open(filename, 'rb'))

            y_pred = classifier.predict(X_test)
            print(classification_report(y_test, y_pred))

            clreport = classification_report(y_test, y_pred)

            print("Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
            print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

            Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
            Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))

            print("Training process is complete Model File Saved!")

            df = df.head(200)

            return render_template('UploadDataSet.html', data=df.to_html(), tacc=Tacc, testacc=Testacc, report=clreport)
        elif typ == "Heart":
            file = request.files['fileupload']
            file_extension = file.filename.split('.')[1]
            print(file_extension)
            # file.save("static/upload/" + secure_filename(file.filename))

            import pandas as pd
            import matplotlib.pyplot as plt
            df = ''
            if file_extension == 'xlsx':
                df = pd.read_excel(file.read(), engine='openpyxl')
            elif file_extension == 'xls':
                df = pd.read_excel(file.read())
            elif file_extension == 'csv':
                df = pd.read_csv(file)

            def clean_dataset(df):
                assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
                df.dropna(inplace=True)
                indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
                return df[indices_to_keep].astype(np.float64)

            df = clean_dataset(df)

            df_copy = df.copy(deep=True)
            df_copy[['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco']] = df_copy[
                ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco']].replace(0,
                                                                                                                  np.NaN)

            # Replacing NaN value by mean, median depending upon distribution

            # df_copy['gender'].fillna(df_copy['gender'].mean(), inplace=True)
            # df_copy['height'].fillna(df_copy['height'].mean(), inplace=True)
            # df_copy['weight'].fillna(df_copy['weight'].mean(), inplace=True)
            # df_copy['ap_hi'].fillna(df_copy['ap_hi'].mean(), inplace=True)
            # df_copy['ap_lo'].fillna(df_copy['ap_lo'].mean(), inplace=True)
            # df_copy['cholesterol'].fillna(df_copy['cholesterol'].mean(), inplace=True)
            # df_copy['gluc'].fillna(df_copy['gluc'].mean(), inplace=True)
            # df_copy['smoke'].fillna(df_copy['smoke'].mean(), inplace=True)
            # df_copy['alco'].fillna(df_copy['alco'].mean(), inplace=True)

            # Model Building
            from sklearn.model_selection import train_test_split
            df.drop(df.columns[np.isnan(df).any()], axis=1)
            X = df.drop(columns='active')
            y = df['active']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

            from sklearn.svm import SVC
            classifier = SVC()
            from sklearn.metrics import classification_report

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            print(classification_report(y_test, y_pred))

            # Creating a pickle file for the classifier
            filename = 'heart-prediction-rfc-model.pkl'
            pickle.dump(classifier, open(filename, 'wb'))

            y_pred = classifier.predict(X_test)
            print(classification_report(y_test, y_pred))

            clreport = classification_report(y_test, y_pred)

            print("Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
            print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

            Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
            Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))

            print("Training process is complete Model File Saved!")

            return render_template('UploadDataSet.html', data=df.to_html(), tacc=Tacc, testacc=Testacc, report=clreport)

        else:
            file = request.files['fileupload']
            file_extension = file.filename.split('.')[1]
            print(file_extension)
            # file.save("static/upload/" + secure_filename(file.filename))

            import pandas as pd
            import matplotlib.pyplot as plt
            df = ''
            if file_extension == 'xlsx':
                df = pd.read_excel(file.read(), engine='openpyxl')
            elif file_extension == 'xls':
                df = pd.read_excel(file.read())
            elif file_extension == 'csv':
                df = pd.read_csv(file)

            df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

            df_copy = df.copy(deep=True)
            df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[
                ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

            # Replacing NaN value by mean, median depending upon distribution
            df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
            df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
            df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
            df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
            df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

            # Model Building
            from sklearn.model_selection import train_test_split
            X = df.drop(columns='Outcome')
            y = df['Outcome']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

            from sklearn.svm import SVC
            from sklearn.metrics import classification_report
            classifier = SVC()

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            print(classification_report(y_test, y_pred))

            # Creating a pickle file for the classifier
            filename = 'diabetes-prediction-rfc-model.pkl'
            pickle.dump(classifier, open(filename, 'wb'))

            y_pred = classifier.predict(X_test)
            print(classification_report(y_test, y_pred))

            clreport = classification_report(y_test, y_pred)

            print("Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
            print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

            Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
            Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))

            print("Training process is complete Model File Saved!")

            return render_template('UploadDataSet.html', data=df.to_html(), tacc=Tacc, testacc=Testacc, report=clreport)


@app.route("/AdminUserInfo")
def AdminUserInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM doctortb ")
    data = cur.fetchall()

    return render_template('AdminUserInfo.html', data=data)


@app.route("/AdminAssignInfo")
def AdminAssignInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM drugtb ")
    data = cur.fetchall()

    return render_template('AdminAssignInfo.html', data=data)


@app.route("/doclogin", methods=['GET', 'POST'])
def doclogin():
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['dname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from ownertb where username='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            data1 = 'Username or Password is wrong'
            return render_template('goback.html', data=data1)


        else:
            print(data[0])
            session['uid'] = data[0]
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM ownertb where username='" + username + "' and Password='" + password + "'")
            data = cur.fetchall()

            return render_template('OwnerHome.html', data=data)


@app.route("/uploadModel", methods=['GET', 'POST'])
def uploadModel():
    if request.method == 'POST':

        name1 = session['dname']
        typ = request.form['typ']
        file = request.files['fileupload']
        file.save("static/upload/" + file.filename)

        secp_k = generate_key()
        privhex = secp_k.to_hex()
        pubhex = secp_k.public_key.format(True).hex()

        filepath = "./static/upload/" + file.filename
        head, tail = os.path.split(filepath)

        newfilepath1 = './static/Encrypt/' + str(tail)
        newfilepath2 = './static/Decrypt/' + str(tail)

        data = 0
        with open(filepath, "rb") as File:
            data = base64.b64encode(File.read())  # convert binary to string data to read file

        print("Private_key:", privhex, "\nPublic_key:", pubhex, "Type: ", type(privhex))

        if (privhex == 'null'):
            alert = 'Please Choose Another File,file corrupted!'
            return render_template('goback.html', data=alert)

        else:

            print("Binary of the file:", data)
            encrypted_secp = encrypt(pubhex, data)
            print("Encrypted binary:", encrypted_secp)

            with open(newfilepath1, "wb") as EFile:
                EFile.write(base64.b64encode(encrypted_secp))
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO  modeltb VALUES ('','" + name1 + "','" + typ + "','" + file.filename + "','" + pubhex + "','" + privhex + "')")
            conn.commit()
            conn.close()
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM modeltb where ownerName='" + name1 + "' ")
            data = cur.fetchall()
            return render_template('UploadInfo.html', data=data)


@app.route("/UploadInfo")
def UploadInfo():
    name1 = session['dname']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM modeltb where ownerName='" + name1 + "' ")
    data = cur.fetchall()
    return render_template('UploadInfo.html', data=data)


@app.route("/UploadModel")
def UploadModel():
    return render_template('UploadModel.html')


@app.route("/RequestInfo")
def RequestInfo():
    name1 = session['dname']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM requesttb where ownerName='" + name1 + "' ")
    data = cur.fetchall()
    return render_template('RequestInfo.html', data=data)


@app.route('/Accept')
def Accept():
    id = request.args.get('id')
    session['id'] = id
    name1 = session['dname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cursor = conn.cursor()
    cursor.execute("SELECT  *  FROM requesttb where  id='" + session['id'] + "'  ")
    data = cursor.fetchone()
    if data:
        username = data[1]
        Prikey = data[5]

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cursor = conn.cursor()
    cursor.execute("SELECT  *  FROM regtb  where  username='" + username + "'  ")
    data = cursor.fetchone()
    if data:
        mail = data[3]
        print(mail)
        sendmail(mail, "Id " + session['id'] + "  Prikey " + str(Prikey))

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cursor = conn.cursor()
    cursor.execute(
        "update  requesttb set status='Accept' where Id='" + session['id'] + "'")
    conn.commit()
    conn.close()

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM requesttb where ownerName='" + name1 + "' ")
    data = cur.fetchall()

    return render_template('RequestInfo.html', data=data)


@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        name1 = request.form['name']
        gender1 = request.form['gender']
        Age = request.form['age']
        email = request.form['email']
        pnumber = request.form['phone']
        address = request.form['address']

        uname = request.form['uname']
        password = request.form['psw']
        loc = request.form['loc']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO regtb VALUES ('" + name1 + "','" + gender1 + "','" + Age + "','" + email + "','" + pnumber + "','" + address + "','" + uname + "','" + password + "','" + loc + "')")
        conn.commit()
        conn.close()
        # return 'file register successfully'

    return render_template('UserLogin.html')


@app.route("/newdoctor", methods=['GET', 'POST'])
def newdoctor():
    if request.method == 'POST':
        name1 = request.form['name']
        gender1 = request.form['gender']
        Age = request.form['age']
        email = request.form['email']
        pnumber = request.form['phone']
        address = request.form['address']
        special = request.form['special']
        loc = request.form['loc']

        uname = request.form['uname']
        password = request.form['psw']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ownertb VALUES ('" + name1 + "','" + gender1 + "','" + Age + "','" + email + "','" + pnumber + "','" + address + "','" + special + "','" + uname + "','" + password + "','" + loc + "')")
        conn.commit()
        conn.close()

    data1 = 'Record Saved'
    return render_template('goback.html', data=data1)


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    error = None
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            data1 = 'Username or Password is Incorrect!'
            return render_template('goback.html', data=data1)



        else:
            print(data[0])
            session['uid'] = data[0]
            mob = data[4]
            email = data[3]

            import random
            n = random.randint(1111, 9999)

            sendmsg(mob, "Your OTP" + str(n))
            sendmail(email, "Your OTP" + str(n))

            session['otp'] = str(n)
            return render_template('OTP.html', data=data)










@app.route("/otplogin", methods=['GET', 'POST'])
def otplogin():
    error = None
    if request.method == 'POST':
        username = request.form['uname']

        if session['otp']==username:

            username1 =  session['uname']
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username1 + "'")
            data = cur.fetchall()

            return render_template('UserHome.html', data=data)

        else:

            data1 = 'OTP is Incorrect!'
            return render_template('goback.html', data=data1)







@app.route("/diabete", methods=['GET', 'POST'])
def newquery():
    if request.method == 'POST':

        Answer = ''
        Prescription = ''

        uname = session['uname']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        skinthickness = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        preg = int(pregnancies)
        glucose = int(glucose)
        bp = int(bloodpressure)
        st = int(skinthickness)
        insul = int(insulin)
        bmi = float(bmi)
        dpf = float(dpf)
        age = int(age)

        filename2 = 'diabetes-prediction-rfc-model.pkl'
        classifier2 = pickle.load(open(filename2, 'rb'))

        data = np.array([[preg, glucose, bp, st, insul, bmi, dpf, age]])
        my_prediction = classifier2.predict(data)
        print(my_prediction)

        if my_prediction == 1:
            Answer = 'Hello:According to our Calculations, You have DIABETES'

            if (glucose <= 120):

                Answer = ' Hello:According to our Calculations, You have  Type1- Diabetic'
                Prescription = 'Managing Glucose in T1D Once Had But One Treatment'
            else:
                Answer = 'Hello:According to our Calculations, You have  Type2- Diabetic'
                Prescription = 'Medications Names:Repaglinide(Prandin)Nateglinide (Starlix)'

            msg = 'Hello:According to our Calculations, You have DIABETES'
            print('Hello:According to our Calculations, You have DIABETES')

            session['Ans'] = 'Yes'
            # Heart Cancer Diabetes
            session['dtype'] = 'Diabetes'


        else:
            Answer = 'Congratulations!!  You DON T have diabetes'

            msg = 'Congratulations!!  You DON T have diabetes'
            print('Congratulations!! You DON T have diabetes')
            Prescription = 'Nill'
            session['Ans'] = 'No'

        return render_template('Answer.html', data=Answer)


@app.route("/heart", methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':

        Answer = ''
        Prescription = ''

        uname = session['uname']

        age = request.form['age']
        gender = request.form['gender']
        height = request.form['height']
        weight = request.form['weight']
        aphi = request.form['aphi']
        aplo = request.form['aplo']
        choles = request.form['choles']
        glucose = request.form['glucose']
        smoke = request.form['smoke']
        alcohol = request.form['alcohol']

        age = int(age)
        gender = int(gender)
        height = int(height)
        weight = int(weight)
        aphi = int(aphi)
        aplo = float(aplo)
        choles = float(choles)
        glucose = int(glucose)
        smoke = int(smoke)
        alcohol = int(alcohol)

        filename2 = 'heart-prediction-rfc-model.pkl'
        classifier2 = pickle.load(open(filename2, 'rb'))

        data = np.array([[age, gender, height, weight, aphi, aplo, choles, glucose, smoke, alcohol]])
        my_prediction = classifier2.predict(data)
        print(my_prediction[0])

        if my_prediction == 1:
            Answer = 'Hello:According to our Calculations, You have Heart disease'

            msg = 'Hello:According to our Calculations, You have Heart disease '
            print('Hello:According to our Calculations, You have Heart disease')

            session['Ans'] = 'Yes'
            # Heart Cancer Diabetes
            session['dtype'] = 'heart'


        else:
            Answer = 'Congratulations!!  You DON T have Heart disease'

            if (aphi >= 100):

                Answer = ' Congratulations!!  You DON T have Heart disease May be Heart Disease Will Affected In Future '
                # Prescription = 'Managing Glucose in T1D Once Had But One Treatment'
            else:
                Answer = 'Congratulations!!  You DON T have Heart disease'
                # Prescription = "Medications Names:Repaglinide(Prandin)Nateglinide (Starlix)"

            msg = 'Congratulations!!  You DON T have Heart disease'
            print('Congratulations!! You DON T have Heart disease')
            Prescription = 'Nill'
            session['Ans'] = 'No'

        return render_template('Answer.html', data=Answer)


@app.route("/cancer", methods=['GET', 'POST'])
def cancer():
    if request.method == 'POST':

        Answer = ''
        Prescription = ''

        uname = session['uname']

        Clumpthickness = request.form['Clumpthickness']
        Uniformityofcellsize = request.form['Uniformityofcellsize']
        Uniformityofcellshape = request.form['Uniformityofcellshape']
        MarginalAdhesion = request.form['MarginalAdhesion']
        SingleEpithelialcellsize = request.form['SingleEpithelialcellsize']
        BareNuclei = request.form['BareNuclei']
        Blandchromatin = request.form['Blandchromatin']
        NormalNucleoli = request.form['NormalNucleoli']
        Mitoses = request.form['Mitoses']

        Clumpthickness = int(Clumpthickness)
        Uniformityofcellsize = int(Uniformityofcellsize)
        Uniformityofcellshape = int(Uniformityofcellshape)
        MarginalAdhesion = int(MarginalAdhesion)
        SingleEpithelialcellsize = int(SingleEpithelialcellsize)
        BareNuclei = int(BareNuclei)
        Blandchromatin = int(Blandchromatin)
        NormalNucleoli = int(NormalNucleoli)
        Mitoses = int(Mitoses)

        filename2 = 'breast-prediction-rfc-model.pkl'
        classifier2 = pickle.load(open(filename2, 'rb'))

        data = np.array([[Clumpthickness, Uniformityofcellsize, Uniformityofcellshape, MarginalAdhesion,
                          SingleEpithelialcellsize, BareNuclei, Blandchromatin, NormalNucleoli, Mitoses]])
        my_prediction = classifier2.predict(data)
        print(my_prediction[0])

        if my_prediction == 2:
            Answer = 'Hello:According to our Calculations, You have Breast Cancer'

            '''if (glucose <= 120):

                Answer = 'Type1- Diabetic'
                Prescription = 'Managing Glucose in T1D Once Had But One Treatment'
            else:
                Answer = 'Type2- Diabetic'
                Prescription = "Medications Names:Repaglinide(Prandin)Nateglinide (Starlix)" '''

            msg = 'Hello:According to our Calculations, You have Breast Cancer '
            print('Hello:According to our Calculations, You have Breast Cancer')

            session['Ans'] = 'Yes'
            # Heart Cancer Diabetes
            session['dtype'] = 'Cancer'


        else:
            Answer = 'Congratulations!!  You DON T have Breast Cancer'

            msg = 'Congratulations!!  You DON T have Breast Cancer'
            print('Congratulations!! You DON T have Breast Cancer')
            Prescription = 'Nill'
            session['Ans'] = 'No'

        return render_template('Answer.html', data=Answer)


@app.route("/UModelInfo", methods=['GET', 'POST'])
def UModelInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM modeltb ")
    data = cur.fetchall()
    return render_template('UModelInfo.html', data=data)


@app.route('/SendRequest')
def SendRequest():
    id = request.args.get('id')
    session['id'] = id

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cursor = conn.cursor()
    cursor.execute("SELECT  *  FROM modeltb where  id='" + session['id'] + "'  ")
    data = cursor.fetchone()

    if data:
        Ownername = data[1]
        type = data[2]
        Model = data[3]
        Privkey = data[5]

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO  requesttb VALUES ('','" + session[
            'uname'] + "','" + Ownername + "','" + type + "','" + Model + "','" + Privkey + "','waiting')")
    conn.commit()
    conn.close()

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM requesttb where UserName='" + session['uname'] + "' ")
    data = cur.fetchall()

    return render_template('URequestInfo.html', data=data)


@app.route('/URequestInfo')
def URequestInfo():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM requesttb where UserName='" + session['uname'] + "' ")
    data = cur.fetchall()

    return render_template('URequestInfo.html', data=data)

@app.route('/Decrypt')
def Decrypt():
    id = request.args.get('id')
    session['id'] = id
    st = request.args.get('st')

    if st=="Accept":
        return render_template('Decrypt.html')
    else:
        data1 = 'Waiting For  Owner Approved! '
        return render_template('goback.html', data=data1)



@app.route("/decryt", methods=['GET', 'POST'])
def decryt():
    if request.method == 'POST':

        keys = request.form['keys']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
        cursor = conn.cursor()
        cursor.execute("SELECT  *  FROM requesttb where  id='" + session['id'] + "' and PriKey='" + keys + "' ")
        data = cursor.fetchone()

        if data:
            typ = data[3]
            prkey = data[5]
            fname = data[4]

            privhex = prkey

            filepath = "./static/Encrypt/" + fname
            head, tail = os.path.split(filepath)

            newfilepath1 = './static/Encrypt/' + str(tail)
            newfilepath2 = './static/Decrypt/' + str(tail)

            data = 0
            with open(newfilepath1, "rb") as File:
                data = base64.b64decode(File.read())

            print(data)
            decrypted_secp = decrypt(privhex, data)
            print("\nDecrypted:", decrypted_secp)
            with open(newfilepath2, "wb") as DFile:
                DFile.write(base64.b64decode(decrypted_secp))

            if typ == "Heart":
                return render_template('Heart.html')
            elif typ == "Breast":
                return render_template('Cancer.html')
            elif typ == "Diabetes":
                return render_template('Diabetes.html')
        else:

            data1 = 'Key Incorrect! '
            return render_template('goback.html', data=data1)


@app.route('/download')
def download():
    id = request.args.get('id')

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
    cursor = conn.cursor()
    cursor.execute("SELECT  *  FROM drugtb where  id = '" + str(id) + "'")
    data = cursor.fetchone()
    if data:
        filename = "static\\upload\\" + data[7]

        return send_file(filename, as_attachment=True)

    else:
        return 'Incorrect username / password !'


@app.route("/search", methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        date = request.form['date']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1Federateddb')
        # cursor = conn.cursor()
        cur = conn.cursor()
        cur.execute("SELECT * FROM assigntb where Lastdate='" + date + "'")
        data = cur.fetchall()

        return render_template('Notification.html', data=data)


def sendmsg(targetno, message):
    import requests
    requests.post(
        "http://smsserver9.creativepoint.in/api.php?username=fantasy&password=596692&to=" + targetno + "&from=FSSMSS&message=Dear user  your msg is " + message + " Sent By FSMSG FSSMSS&PEID=1501563800000030506&templateid=1507162882948811640")


def sendmail(Mailid, message):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "sampletest685@gmail.com"
    toaddr = Mailid

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = message

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "hneucvnontsuwgpj")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
