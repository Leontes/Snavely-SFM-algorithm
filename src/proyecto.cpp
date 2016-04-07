#include "proyecto.h"
#include <opencv2/nonfree/nonfree.hpp>
#include <cvsba/cvsba.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <utility>

using namespace std;
using namespace cv;
using namespace cvsba;



/**
Pinta una imagen
*/
void pintaI(Mat im) {
    namedWindow("Proyecto", 1);
    imshow("Proyecto", im);
    waitKey(0);
    destroyWindow("Proyecto 3");
}

/**
Lee las imagenes necesarias para el proyecto
*/
void leeImagenes(const char *filename, vector<Mat> &imagenes) {
    Mat img = imread(filename, 1);
    imagenes.push_back(img);
}

void leeMatrices(const char *filename, vector <Mat> &Ks) {
    Mat aux (3, 3, CV_64F);
    double d;
    ifstream fichero(filename);
    if (fichero) {
        fichero >> d;
        aux.at<double>(0, 0) = d;
        fichero >> d;
        aux.at<double>(0, 1) = d;
        fichero >> d;
        aux.at<double>(0, 2) = d;
        fichero >> d;
        aux.at<double>(1, 0) = d;
        fichero >> d;
        aux.at<double>(1, 1) = d;
        fichero >> d;
        aux.at<double>(1, 2) = d;
        fichero >> d;
        aux.at<double>(2, 0) = d;
        fichero >> d;
        aux.at<double>(2, 1) = d;
        fichero >> d;
        aux.at<double>(2, 2) = d;
        fichero.close();
        Ks.push_back(aux);
    }
}
/**
Ejecuta el detector de caracteristicas SIFT sobre la imagen seleccionada, alternativamente tambien pinta la
imagen con los puntos calculados
*/
void sift(const Mat &img, vector<KeyPoint> &keypoints, int modo) {
    SIFT detector = SIFT(0, 3, 0.04, 10, 1.6);

    //Creamos una matriz mascara
    Mat mask;
    //Calculamos los puntos y los guardamos en el vector de keypoints
    detector.operator()(img, mask, keypoints);

    //Codigo extra para mostrar los puntos por pantalla
    if (modo == 1) {
        Mat salida;
        drawKeypoints(img, keypoints, salida);
        pintaI(salida);
    }
}
/**
Calcula las caracteristicas para todas las imagenes de un array
*/
void computeSift(const vector<Mat> &imagenes, vector<vector<KeyPoint> > &keypoints) {
    vector<KeyPoint> kp;
    keypoints.resize(imagenes.size());
    int cont = 0;
    #pragma omp parallel for private(kp) shared(imagenes, keypoints,cont)
    for (unsigned int i = 0; i < imagenes.size(); i++) {
        sift(imagenes[i], kp, 0);
        #pragma omp critical
        {
            keypoints[i] = kp;
            cout << "Calculo de los puntos SIFT terminado(" << cont + 1 << "/" << keypoints.size() << ")" << endl;
            cont ++;
        }
    }
}

/**
Funcion que calcula los emparejamientos de los puntos de 2 imagenes dadas usando el emparejador de fuerza bruta
*/
void match(const Mat &img1, vector<KeyPoint> &keypoints1, const Mat &img2, vector<KeyPoint> &keypoints2, vector<DMatch> &matches ) {
    //Creamos el extractor de emparejamientos de puntos SIFT
    SiftDescriptorExtractor extractor;

    //Creamos 2 matrices para guardar los descriptores de cada imagen
    Mat descriptors1, descriptors2;

    //Calculamos los descriptores
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    //Calculamos los emparejamiento de cada descriptor mediante un emparejador de fuerza bruta
    bool crossCheck = true;
    BFMatcher matcher (NORM_L2, crossCheck );
    //FlannBasedMatcher matcher;
    matcher.match(descriptors1, descriptors2, matches);

}
/**
Calcula las correspondencias para todas las imagenes de un array
*/
void computeMatches(const vector<Mat> &imagenes, vector<vector<KeyPoint> > &keypoints, vector<vector<DMatch> > &matches, vector<Matchs> &matchs) {
    vector<DMatch> mtch;

    int nMatches = 0;

    for (unsigned int i = 1; i < imagenes.size(); i++) {
        nMatches += i;
    }

    matches.resize(nMatches);

    int cont = 0;

    #pragma omp parallel for private(mtch) shared(imagenes, keypoints, matches, matchs, cont)
    for (unsigned int i = 0; i < imagenes.size(); i++) {
        for (unsigned int j = (i + 1); j < imagenes.size(); j++) {
            match(imagenes[i], keypoints[i], imagenes[j], keypoints[j], mtch);

            #pragma omp critical
            {
                matches[cont] = mtch;
                matchs.push_back(Matchs(i, i, j, j, cont));
                cout << "Calculo de correspondencias terminado(" << cont + 1 << "/" << nMatches << ")" << endl;
                cont ++;
            }


        }
    }
}
/**
Filtra las correspondencias generadas previamente segun la tesis de Snavely. Calcula la matriz F y elimina aquellas correspondencias
que no se encuentran en las lineas epipolares de la imagen
*/
Mat matchesFilter(vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches) {

    Mat mask;
    vector< Point2d > pts1;
    vector< Point2d > pts2;

    for ( unsigned int i = 0; i < matches.size(); i++ ) {
        pts1.push_back(keypoints1[ matches[i].queryIdx ].pt );
        pts2.push_back(keypoints2[ matches[i].trainIdx ].pt);
    }

    Mat F = findFundamentalMat(pts1, pts2, CV_FM_RANSAC + CV_FM_8POINT, 1, 0.9999, mask);

    vector<DMatch> aux;
    Mat x1 (3, 1, CV_64F);
    Mat x2 (1, 3, CV_64F);
    Mat resultado (1, 1, CV_64F);
    for ( unsigned int i = 0; i < pts1.size(); i++ ) {
        x1.at<double> (0, 0) = (double) pts1[i].x;
        x1.at<double> (1, 0) = (double) pts1[i].y;
        x1.at<double> (2, 0) = (double) 1.0;

        x2.at<double> (0, 0) = (double) pts2[i].x;
        x2.at<double> (0, 1) = (double) pts2[i].y;
        x2.at<double> (0, 2) = (double) 1.0;
        resultado = (x2 * F * x1);
        if (resultado.at<double>(0, 0) < 0.005 && resultado.at<double>(0, 0) > -0.005) {
            aux.push_back(matches[i]);
        }
    }
    matches = aux;
    return F;
}
/**
Funcion auxiliar para la comparacion de Keypoints
*/
bool compareKps(KeyPoint kp1, KeyPoint kp2) {
    if (kp1.pt.x != kp2.pt.x) {
        return false;
    }
    else {
        if (kp1.pt.y != kp2.pt.y) {
            return false;
        }
        else {
            return true;
        }
    }
}
/**
Devuelve verdadero si se encuentra un Keypoint dentro de una lista de <imagenes,Keypoints>
*/
bool found(vector<pair< unsigned int, KeyPoint> > listPts, unsigned int img, KeyPoint kp) {
    bool encontrado = false;
    for (unsigned int i = 0; i < listPts.size(); i++) {
        if (listPts[i].first == img || compareKps(listPts[i].second, kp)) {
            encontrado = true;
        }
    }
    return encontrado;
}

/**
Toma todas las correspondencias de una pareja de imagenes, y agrupa aquellas que apunten al mismo punto.
Aunque pertenezcan a otra serie de imagenes.
*/
void includeCorrespondences(listaCorrespondencias &list, vector<DMatch> &matches, unsigned int img1, unsigned int img2, vector<KeyPoint> kps1, vector<KeyPoint> kps2) {
    bool incluido = false;
    //Para todos los matches
    for (unsigned int i = 0; i < matches.size(); i++) {
        //Tomamos una lista de puntos
        for (unsigned int j = 0; j < list.size(); j++) {
            //Miramos punto por punto de la lista
            for (unsigned int k = 0; k < list[j].size(); k++) {
                //cout << i <<" "<<j<<" "<< k<<endl;
                //Si alguna de las imagenes coincide
                if (img1 == list[j][k].first) {
                    //Y ademas coincide el keypoint
                    if (compareKps(kps1[matches[i].queryIdx], list[j][k].second) == true) {
                        //Incluimos el contrario
                        if (found(list[j], img2, kps2[matches[i].trainIdx]) == false) {
                            list[j].push_back(pair<unsigned int, KeyPoint >(img2, kps2[matches[i].trainIdx]));
                            incluido = true;
                        }
                    }
                }
                //Igual que el anterior con en el otro caso
                if (img2 == list[j][k].first) {
                    if (compareKps(kps2[matches[i].trainIdx], list[j][k].second) == true) {
                        if (found(list[j], img1, kps1[matches[i].queryIdx]) == false) {
                            list[j].push_back(pair< unsigned int, KeyPoint >(img1, kps1[matches[i].queryIdx]));
                            incluido = true;
                        }
                    }
                }
            }
        }
        //Si hemos recorrido toda la lista de puntos sin incluir nada, metemos el match completo
        if (incluido == false) {
            list.push_back(vector< pair<unsigned int, KeyPoint > > ());
            list[list.size() - 1].push_back(pair<unsigned int, KeyPoint >(img1, kps1[matches[i].queryIdx]));
            list[list.size() - 1].push_back(pair<unsigned int, KeyPoint >(img2, kps2[matches[i].trainIdx]));

        }
        else {
            incluido = false;
        }
    }
}

/**
Toma todas las parejas de correspondencias y calcula las correspondencias entre todas ellas
*/
void computeCorrespondences(vector<vector<DMatch> > &matches, vector<vector<KeyPoint> > &keypoints, vector<Matchs> &matchs, listaCorrespondencias &list) {
    vector<DMatch> aux;
    vector<KeyPoint> kps1, kps2;
    for (unsigned int i = 0; i < matchs.size(); i++) {
        if (matchs[i].isValid()) {
            aux = matches[matchs[i].matches];
            kps1 = keypoints[matchs[i].kps1];
            kps2 = keypoints[matchs[i].kps2];
            includeCorrespondences(list, aux, matchs[i].img1, matchs[i].img2, kps1, kps2);
        }
    }
}

/**
Evalua la distancia entre las parejas de correspondencias segun el criterio de Snavely
*/
void evaluateMatch(Size imSize, vector<vector<DMatch> > &matches, vector<vector<KeyPoint> > &keypoints, Matchs &match) {
    vector<KeyPoint> Kp1, Kp2;
    vector<DMatch> matchs;
    Kp1 = keypoints[match.kps1];
    Kp2 = keypoints[match.kps2];
    matchs = matches[match.matches];

    vector< Point2d > pts1;
    vector< Point2d > pts2;

    for ( unsigned int i = 0; i < matchs.size(); i++ ) {
        pts1.push_back(Kp1[ matchs[i].queryIdx ].pt );
        pts2.push_back(Kp2[ matchs[i].trainIdx ].pt);
    }
    double threshold = imSize.width;
    if (imSize.height > threshold) {
        threshold = imSize.height;
    }
    Mat mask;
    //Calcula la homografia asociada a las 2 imagenes
    Mat H = findHomography(pts1, pts2, CV_RANSAC, 0.4 * threshold, mask);

    int cont = 0;
    //Si tenemos pocas correspondencias descartamos todas
    if (matchs.size() < 100) {
        match.percent = -1.0;
    }
    else {
        //Si no, vemos que porcentaje de correspondencias que se encuentran en la imagen
        //mask[i]==0
        for (int i = 0; i < mask.rows; i++) {
            cont += (int)mask.at<unsigned char>(i, 0);
        }
        match.percent = (double)cont / (double)mask.rows;
    }
}

/**
Calculamos la matriz esencial segun el metodo de Harley & Zisserman
*/
Mat calcEssentialMat(Mat K, Mat K1, Mat F) {
    return K1.t() * F * K;
}

/**
Descomponemos la matriz esencial segun el metodo de descomposicion
de valores singulares de Harley & Zisserman
*/
void decomposeE(Mat E, Mat &R, Mat &T) {
    SVD svd(E);
    Matx33d W(0, -1, 0,
              1, 0, 0,
              0, 0, 1);
    Matx33d Winv(0, 1, 0,
                 -1, 0, 0,
                 0, 0, 1);
    R = svd.u * Mat(W) * svd.vt;
    T = svd.u.col(2);
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 Codigo inspirado en el encontrado en www.morethantechnical.com
 */
Mat LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
                          Matx34d P,       //camera 1 matrix
                          Point3d u1,      //homogenous image point in 2nd camera
                          Matx34d P1       //camera 2 matrix
                         )
{
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    Matx43d A(u.x * P(2, 0) - P(0, 0),    u.x * P(2, 1) - P(0, 1),      u.x * P(2, 2) - P(0, 2),
              u.y * P(2, 0) - P(1, 0),    u.y * P(2, 1) - P(1, 1),      u.y * P(2, 2) - P(1, 2),
              u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1),   u1.x * P1(2, 2) - P1(0, 2),
              u1.y * P1(2, 0) - P1(1, 0), u1.y * P1(2, 1) - P1(1, 1),   u1.y * P1(2, 2) - P1(1, 2)
             );
    Matx41d B(-(u.x * P(2, 3) - P(0, 3)),
              -(u.y * P(2, 3)  - P(1, 3)),
              -(u1.x * P1(2, 3)    - P1(0, 3)),
              -(u1.y * P1(2, 3)    - P1(1, 3)));

    Mat X;
    solve(A, B, X, DECOMP_SVD);

    return X;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 Codigo inspirado en el encontrado en www.morethantechnical.com
 */
Mat IterativeLinearLSTriangulation(Point3d u,    //homogenous image point (u,v,1)
                                   Matx34d P,          //camera 1 matrix
                                   Point3d u1,         //homogenous image point in 2nd camera
                                   Matx34d P1          //camera 2 matrix
                                  ) {
    double wi = 1, wi1 = 1, EPSILON = 0.5;
    Mat_<double> X(4, 1);
    for (int i = 0; i < 10; i++) { //Hartley suggests 10 iterations at most
        Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;

        //recalculate weights
        double p2x = Mat_<double>(Mat_<double>(P).row(2) * X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2) * X)(0);

        //breaking point
        if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        Matx43d A((u.x * P(2, 0) - P(0, 0)) / wi,       (u.x * P(2, 1) - P(0, 1)) / wi,         (u.x * P(2, 2) - P(0, 2)) / wi,
                  (u.y * P(2, 0) - P(1, 0)) / wi,       (u.y * P(2, 1) - P(1, 1)) / wi,         (u.y * P(2, 2) - P(1, 2)) / wi,
                  (u1.x * P1(2, 0) - P1(0, 0)) / wi1,   (u1.x * P1(2, 1) - P1(0, 1)) / wi1,     (u1.x * P1(2, 2) - P1(0, 2)) / wi1,
                  (u1.y * P1(2, 0) - P1(1, 0)) / wi1,   (u1.y * P1(2, 1) - P1(1, 1)) / wi1,     (u1.y * P1(2, 2) - P1(1, 2)) / wi1
                 );
        Mat_<double> B = (Mat_<double>(4, 1) <<    -(u.x * P(2, 3)    - P(0, 3)) / wi,
                          -(u.y * P(2, 3)  - P(1, 3)) / wi,
                          -(u1.x * P1(2, 3)    - P1(0, 3)) / wi1,
                          -(u1.y * P1(2, 3)    - P1(1, 3)) / wi1
                         );

        solve(A, B, X_, DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;
    }
    return X;
}


/**
Funcion que triangula todos los puntos de un par de imagenes
Codigo inspirado en el encontrado en www.morethantechnical.com

*/
void TriangulatePoints(const vector<KeyPoint> &Kp1, const vector<KeyPoint> &Kp2, const vector<DMatch> &matches, const Mat &K, const Matx34d &P, const Matx34d &P1, vector<Point3d> &pointcloud, vector<KeyPoint> &correspImg1Pt) {
    vector<double> depths;
    pointcloud.clear();
    correspImg1Pt.clear();
    Matx44d P1_(P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
                P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
                P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3),
                0,      0,      0,      1);
    Matx44d P1inv(P1_.inv());

    cout << "Triangulando..." << endl;
    Mat_<double> KP1 = K * Mat(P1);
    #pragma omp parallel for shared(pointcloud, correspImg1Pt)
    //Tomamos 2 puntos
    for (unsigned int i = 0; i < matches.size(); i++) {
        Point2d kp = Kp1[matches[i].queryIdx].pt;
        Point3d u(kp.x, kp.y, 1.0);
        Mat_<double> um = K * Mat_<double>(u);
        u.x = um(0); u.y = um(1); u.z = um(2);

        Point2d kp1 = Kp2[matches[i].trainIdx].pt;
        Point3d u1(kp1.x, kp1.y, 1.0);
        Mat_<double> um1 = K * Mat_<double>(u1);
        u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);
        //Se los pasamos a la funcion que calcula la triangulacion
        Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);

        Mat_<double> xPt_img = KP1 * X;             //reproject
        Point2d xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

        #pragma omp critical
        {
            //Guardamos el resultado
            Point3d cp;
            cp = Point3d(X(0), X(1), X(2));

            pointcloud.push_back(cp);
            correspImg1Pt.push_back(Kp1[i]);
            depths.push_back(X(2));
        }
    }
    cout << "Terminado" << endl;

}

/**
Funcion que reajusta los puntos ya calculados a partir de unas nuevas correspondencias
*/
void reAdjustPoints(vector<Point3d> pointcloudReadjust, vector< vector< Point2d > > pointsImg , listaCorrespondencias list, unsigned int img1, unsigned int img2, Mat K, Mat K1, Matx34d P, Matx34d P1) {
    vector< vector< int > > visibility;
    vector< Mat > cameraMatrix, distCoeffs, R, T;
    pointsImg.resize(2);


    // Todos los puntos son visibles
    visibility.resize(2);
    for (unsigned int i = 0; i < 2; i++)  {
        visibility[i].resize(pointcloudReadjust.size());
        for (unsigned int j = 0; j < pointcloudReadjust.size(); j++) {
            visibility[i][j] = 1;
        }
    }
    // Fijamos los valores intrinsecos de las camaras
    cameraMatrix.resize(2);
    cameraMatrix[0] = K;
    cameraMatrix[1] = K1;

    // Asumimos que no tenemos distorsion en las camaras
    distCoeffs.resize(2);
    for (int i = 0; i < 2; i++) {
        distCoeffs[i] = cv::Mat(5, 1, CV_64FC1, cv::Scalar::all(0));
    }

    // Fijamos la rotacion de las camaras(La primera esta fija)
    R.resize(2);
    R[0] = Mat(P);
    R[0] = R[0](Rect(0, 0, 3, 3));
    R[1] = Mat(P1);
    R[1] = R[1](Rect(0, 0, 3, 3));

    // Fijamos la translacion de las camaras(La primera esta fija)
    T.resize(2);
    T[0] = Mat(P);
    T[0] = T[0](Rect(3, 0, 1, 3));
    T[1] = Mat(P1);
    T[1] = T[1](Rect(3, 0, 1, 3));

    //Reajustamos
    Sba sba;
    sba.run(pointcloudReadjust,  pointsImg,  visibility,  cameraMatrix,  R,  T, distCoeffs);

    cout << "Error antes del reajuste =" << sba.getInitialReprjError() << ". Error final =" << sba.getFinalReprjError() << endl;
}

/**
Calcula la triangulacion para TODAS las imagenes dadas
*/
void computeTriangulations(vector<vector<KeyPoint> > &keypoints, vector<vector<DMatch> > &matches, vector<Matchs> &matchs, listaCorrespondencias list, vector<Mat> &Ks, vector<Matx34d> &Ps, vector<Point3d> &pointcloud, vector<KeyPoint> &correspImg1Pt) {
    cout << "Tomando las imagenes " << matchs[0].img1 << " y " << matchs[0].img2 << " como par inicial" << endl;
    Mat E = calcEssentialMat(Ks[matchs[0].img1], Ks[matchs[0].img2], matchs[0].F);
    Mat R, T;
    decomposeE(E, R, T);

    Matx34d P(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    Ps.push_back(P);

    Mat P1;
    hconcat(R, T, P1);
    Ps.push_back(P1);

    vector<Point3d> pointcloudAux, pointcloudReadjust;
    vector<KeyPoint> correspImg1PtAux;
    vector<vector<Point2d> > correspImg1PtReadjust(2);

    //Triangula el primer par de imagenes
    TriangulatePoints(keypoints[matchs[0].kps1], keypoints[matchs[0].kps2], matches[matchs[0].matches], Ks[matchs[0].img2], P, P1, pointcloud, correspImg1Pt);
    bool insertar;
    for (unsigned int i = 1; i < matchs.size(); i++) {
        //Si las correspondencias son validas
        if (matchs[i].isValid()) {
            //Comprueba que la imagen a reconstruir tenga correspondencias
            if (matchs[i].img1 == matchs[0].img1) {
                E = calcEssentialMat(Ks[matchs[0].img1], Ks[matchs[i].img2], matchs[i].F);
                hconcat(R, T, P1);
                Ps.push_back(P1);
                //Triangula puntos para el siguiente par de imagenes (una fija, el resto rotan a partir de ella)
                TriangulatePoints(keypoints[matchs[0].kps1], keypoints[matchs[i].kps2], matches[matchs[i].matches], Ks[matchs[i].img2], P, P1, pointcloudAux, correspImg1PtAux);
                //Comprobamos que los puntos calculados no hallan sido previamente incluidos en la nube de puntos
                for (unsigned j = 0; j < correspImg1PtAux.size(); j++) {
                    insertar = true;
                    for (unsigned int k = 0; k < correspImg1Pt.size(); k++) {
                        if (compareKps(correspImg1PtAux[j], correspImg1Pt[k])) {
                            insertar = false;
                        }
                    }
                    //Si no estan incluidos, se incluyen
                    if (insertar == true) {
                        pointcloud.push_back(pointcloudAux[j]);
                        correspImg1Pt.push_back(correspImg1PtAux[j]);

                    }
                    //Si ya estaban incluyen para recalcularlos
                    else {
                        pointcloudReadjust.push_back(pointcloudAux[j]);
                        correspImg1PtReadjust[0].push_back(correspImg1PtAux[j].pt);
                        correspImg1PtReadjust[1].push_back(keypoints[matchs[i].kps2][j].pt);
                    }
                }
                //Si tenemos puntos que reajustar, los reajustamos
                if (pointcloudReadjust.size() != 0) {
                    cout << "Reajustando puntos... " << endl;
                    reAdjustPoints(pointcloudReadjust, correspImg1PtReadjust, list, matchs[0].img1, matchs[i].img2, Ks[matchs[0].img1], Ks[matchs[i].img2], P, P1);
                    for (unsigned int k = 0; k < pointcloudReadjust.size(); k++) {
                        for (unsigned int l = 0; l < pointcloud.size(); l++) {
                            if (correspImg1PtReadjust[0][k].x == correspImg1Pt[l].pt.x) {
                                if (correspImg1PtReadjust[0][k].y == correspImg1Pt[l].pt.y) {
                                    pointcloud[l] = pointcloudReadjust[k];
                                }
                            }
                        }
                    }
                    pointcloudReadjust.clear();
                    correspImg1PtReadjust[0].clear();
                    correspImg1PtReadjust[1].clear();
                }

            }
            //Igual que el bloque anterior
            if (matchs[i].img2 == matchs[0].img1) {
                E = calcEssentialMat(Ks[matchs[0].img1], Ks[matchs[i].img1], matchs[i].F);
                hconcat(R, T, P1);
                Ps.push_back(P1);
                TriangulatePoints(keypoints[matchs[0].kps1], keypoints[matchs[1].kps1], matches[matchs[i].matches], Ks[matchs[i].img1], P1, P, pointcloudAux, correspImg1PtAux);
                for (unsigned j = 0; j < correspImg1PtAux.size(); j++) {
                    insertar = true;
                    for (unsigned int k = 0; k < correspImg1Pt.size(); k++) {
                        if (compareKps(correspImg1PtAux[j], correspImg1Pt[k])) {
                            insertar = false;
                        }
                    }
                    if (insertar == true) {
                        pointcloud.push_back(pointcloudAux[j]);
                        correspImg1Pt.push_back(correspImg1PtAux[j]);
                    }
                    else {
                        pointcloudReadjust.push_back(pointcloudAux[j]);
                        correspImg1PtReadjust[0].push_back(correspImg1PtAux[j].pt);
                        correspImg1PtReadjust[1].push_back(keypoints[matchs[i].kps2][j].pt);
                    }
                }

                if (pointcloudReadjust.size() != 0) {
                    cout << "Reajustando puntos..." << endl;
                    reAdjustPoints(pointcloudReadjust, correspImg1PtReadjust, list, matchs[0].img1, matchs[i].img1, Ks[matchs[0].img1], Ks[matchs[i].img1], P, P1);
                    for (unsigned int k = 0; k < correspImg1PtReadjust.size(); k++) {
                        for (unsigned int l = 0; l < correspImg1Pt.size(); l++) {
                            if (correspImg1PtReadjust[0][k].x == correspImg1Pt[l].pt.x) {
                                if (correspImg1PtReadjust[0][k].y == correspImg1Pt[l].pt.y) {
                                    pointcloud[l] = pointcloudReadjust[k];
                                }
                            }
                        }
                    }
                    pointcloudReadjust.clear();
                    correspImg1PtReadjust[0].clear();
                    correspImg1PtReadjust[1].clear();
                }

            }

        }
    }

}

/**
Funcion que pinta la imagen con los puntos triangulados, optativamente tambien la guarda en el HD
*/
void DepthIm(Size imSize, vector<Point3d> &pointcloud, vector<KeyPoint> &correspImg1Pt) {
    double minVal, maxVal;
    vector<double> depths;
    for (unsigned int i = 0; i < pointcloud.size(); i++) {
        depths.push_back(pointcloud[i].z);
    }

    minMaxLoc(depths, &minVal, &maxVal);
    Mat tmp(imSize, CV_8UC3, Scalar(0, 0, 0)); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
    for (unsigned int i = 0; i < pointcloud.size(); i++) {
        double _d = MAX(MIN((pointcloud[i].z - minVal) / (maxVal - minVal), 1.0), 0.0);
        circle(tmp, correspImg1Pt[i].pt, 1, Scalar(255 * (1.0 - (_d)), 255, 255), CV_FILLED);
    }

    pintaI(tmp);

    //imwrite( "puntos.jpg", tmp );
}