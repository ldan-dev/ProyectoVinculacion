# ProyectoVinculacion

<br /> <div align="center"> <a href=""> <img src="https://github.com/user-attachments/assets/19d47f04-ba88-4e29-8737-3a8e8f39a6cd" alt="Logo UG" width="250"> </a> <h3 align="center">Student Performance Predictor</h3> <p align="center"> Bachelor’s Degree in Data Engineering and Artificial Intelligence (IS75LI0801)<br /> Universidad de Guanajuato – Campus Irapuato-Salamanca </p> </div>

Access the application here
Contact Information

    ld.avinaneri@ugto.mx
    daniel.avina.neri@gmail.com
    GitHub Profile

Project Overview

student_performance is a web application designed to predict student performance using a linear regression model based on key academic and lifestyle factors. The project is developed as part of the Bachelor’s Degree in Data Engineering and Artificial Intelligence at Universidad de Guanajuato, Campus Irapuato-Salamanca.
Model Information

The predictive model incorporates the following features:

    Hours Studied
    Previous Scores
    Extracurricular Activities
    Sleep Hours
    Sample Question Papers Practiced

Model coefficients:
JSON

{
    "Hours Studied": 2.85,
    "Previous Scores": 1.02,
    "Extracurricular Activities": 0.61,
    "Sleep Hours": 0.48,
    "Sample Question Papers Practiced": 0.19,
    "Intercept": -34.08
}

Model Evaluation

    Mean Squared Error (MSE): 4.1514
    Indicates the average squared difference between predicted and actual values. Lower values signify higher accuracy.
    Coefficient of Determination (R²): 0.9888
    Represents the proportion of variance explained by the model. Values close to 1 denote excellent model fit.

Usage

Users can input relevant data (study hours, previous scores, extracurricular activities, sleep hours, and sample papers practiced) to forecast their expected performance. The application provides real-time predictions and interprets the results for better understanding.
File Descriptions

    index.html: Main web interface for data input and result visualization.
    styles.css: Custom styling for the application’s layout and components.
    app.js: Handles form logic, prediction calculations, and result visualization.
    model/predict.js: (if present) Contains the linear regression model logic used for predictions.
    README.md: Documentation and project overview.
    assets/: Contains images and other static resources.

License & Attribution

Developed by Leonardo Daniel Aviña Neri
Program: LIDIA (Data Engineering and Artificial Intelligence)
Course: Probability
Universidad de Guanajuato – Campus Irapuato-Salamanca

<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fugto.mx%2Fcampusirapuatosalamanca%2Fimages%2Fescudos%2Fdicis_png.png&f=1&nofb=1&ipt=b573c2bbbf60b482c67403316aed4ffa2f6c6c8eb42b7377665b7912c045aefb&ipo=images" alt="Logo DICIS UG" style="display: block; margin: 10px auto; max-width: 550px; height: auto;">
