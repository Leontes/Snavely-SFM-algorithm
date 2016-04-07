#ifndef PROYECTO_H
#define PROYECTO_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


typedef vector< vector < pair < unsigned int, KeyPoint > > > listaCorrespondencias;

struct  Matchs {

    unsigned int img1;
    unsigned int img2;
    unsigned int kps1;
    unsigned int kps2;
    unsigned int matches;
    Mat F;
    double percent;

    bool valido;

    Matchs(unsigned int imagen1, unsigned int keypoints1, unsigned int imagen2, unsigned int keypoints2, unsigned int mtchs) {
        img1 = imagen1;
        img2 = imagen2;
        kps1 = keypoints1;
        kps2 = keypoints2;
        matches = mtchs;
        valido = true;
        percent = 0.0;
    }

    Matchs() {
        img1 = 0;
        img2 = 0;
        kps1 = 0;
        kps2 = 0;
        matches = 0;
        valido = false;
        percent = 0.0;
    }

    void setInvalid() {
        valido = false;
    }

    bool isValid() {
        return valido;
    }
    void setF(Mat F1) {
        F1.copyTo(F);
    }
    Matchs &operator = (Matchs mt) {
        if (this != &mt) {
            img1 = mt.img1;
            img2 = mt.img2;
            kps1 = mt.kps1;
            kps2 = mt.kps2;
            matches = mt.matches;
            valido = mt.valido;
            percent = mt.percent;
        }
        return *this;
    }
};


//Funciones auxiliares
void pintaI(Mat im);
void leeImagenes(const char *filename, vector<Mat> &imagenes);
void leeMatrices(const char *filename, vector <Mat> &Ks);

//Funciones de calculo de caracteristicas
void sift(const Mat &img, vector<KeyPoint> &keypoints, int modo);
void computeSift(const vector<Mat> &imagenes, vector<vector<KeyPoint> > &keypoints);

//Funciones para el calculo y tratamiento de correspondencias
void match(const Mat &img1, vector<KeyPoint> &keypoints1, const Mat &img2, vector<KeyPoint> &keypoints2, vector<DMatch> &matches );
void computeMatches(const vector<Mat> &imagenes, vector<vector<KeyPoint> > &keypoints, vector<vector<DMatch> > &matches, vector<Matchs> &matchs);
Mat matchesFilter(vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches);

//Funciones para calcular las correspondencias entre imagenes
void includeCorrespondences(listaCorrespondencias &list, vector<DMatch> &matches, unsigned int img1, unsigned int img2, vector<KeyPoint> kps1, vector<KeyPoint> kps2);
void computeCorrespondences(vector<vector<DMatch> > &matches, vector<vector<KeyPoint> > &keypoints, vector<Matchs> &matchs, listaCorrespondencias &list);

//Funciones para el calculo de las matrices referentes a los parametros de la camara
Mat calcEssentialMat(Mat K, Mat K1, Mat F);
void decomposeE(Mat E, Mat &R, Mat &T);
void evaluateMatch(Size imSize, vector<vector<DMatch> > &matches, vector<vector<KeyPoint> > &keypoints, Matchs &match);

//Funciones para el calculo de la triangulacion de puntos
Mat IterativeLinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1);
void TriangulatePoints(const vector<KeyPoint> &Kp1, const vector<KeyPoint> &kp2, const vector<DMatch> &matches, const Mat &K, const Matx34d &P, const Matx34d &P1, vector<Point3d> &pointcloud, vector<KeyPoint> &correspImg1Pt);
void computeTriangulations(vector<vector<KeyPoint> > &keypoints, vector<vector<DMatch> > &matches, vector<Matchs> &matchs, listaCorrespondencias list, vector<Mat> &Ks, vector<Matx34d> &Ps, vector<Point3d> &pointcloud, vector<KeyPoint> &correspImg1Pt);

//Funciones para el calculo del reajuste de puntos ya calculados previamente
void getCorrespondencePoints(listaCorrespondencias list, vector<Point3d> &pointcloud, vector<Point3d> &pointcloudReadjust, vector<KeyPoint> &correspImg1Pt, vector<KeyPoint> &correspImg1PtReadjust);
void reAdjustPoints(vector<Point3d> pointcloudReadjust, vector< vector< Point2d > > pointsImg, listaCorrespondencias list, unsigned int img1, unsigned int img2, Mat K, Mat K1, Matx34d P, Matx34d P1);

//Funcion para guardar la imagen de profundidad
void DepthIm(Size imSize, vector<Point3d> &pointcloud, vector<KeyPoint> &correspImg1Pt);

#endif