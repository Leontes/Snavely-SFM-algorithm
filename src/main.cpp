#include "proyecto.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


bool compararMatchs (Matchs mt1, Matchs mt2) {
    if (mt1.percent == -1) {
        return false;
    }
    if (mt2.percent == -1) {
        return true;
    }
    return mt1.percent < mt2.percent;
}

int main(int argc, char *argv[]) {
    vector<Mat> Ks;

    vector<Mat> imagenes;


    leeImagenes("imagenes/0000.png", imagenes);
    leeImagenes("imagenes/0001.png", imagenes);
    leeImagenes("imagenes/0002.png", imagenes);
    leeImagenes("imagenes/0003.png", imagenes);
    /*leeImagenes("imagenes/0004.png", imagenes);
    leeImagenes("imagenes/0005.png", imagenes);
    leeImagenes("imagenes/0006.png", imagenes);
    leeImagenes("imagenes/0007.png", imagenes);
    leeImagenes("imagenes/0008.png", imagenes);
    leeImagenes("imagenes/0009.png", imagenes);
    leeImagenes("imagenes/0010.png", imagenes);
    leeImagenes("imagenes/0011.png", imagenes);
    leeImagenes("imagenes/0012.png", imagenes);
    leeImagenes("imagenes/0013.png", imagenes);
    leeImagenes("imagenes/0014.png", imagenes);
    leeImagenes("imagenes/0015.png", imagenes);
    leeImagenes("imagenes/0016.png", imagenes);
    leeImagenes("imagenes/0017.png", imagenes);
    leeImagenes("imagenes/0018.png", imagenes);*/

    leeMatrices("imagenes/0000.png.K", Ks);
    leeMatrices("imagenes/0001.png.K", Ks);
    leeMatrices("imagenes/0002.png.K", Ks);
    leeMatrices("imagenes/0003.png.K", Ks);
    /*leeMatrices("imagenes/0004.png.K", Ks);
    leeMatrices("imagenes/0005.png.K", Ks);
    leeMatrices("imagenes/0006.png.K", Ks);
    leeMatrices("imagenes/0007.png.K", Ks);
    leeMatrices("imagenes/0008.png.K", Ks);
    leeMatrices("imagenes/0009.png.K", Ks);
    leeMatrices("imagenes/0010.png.K", Ks);
    leeMatrices("imagenes/0011.png.K", Ks);
    leeMatrices("imagenes/0012.png.K", Ks);
    leeMatrices("imagenes/0013.png.K", Ks);
    leeMatrices("imagenes/0014.png.K", Ks);
    leeMatrices("imagenes/0015.png.K", Ks);
    leeMatrices("imagenes/0016.png.K", Ks);
    leeMatrices("imagenes/0017.png.K", Ks);
    leeMatrices("imagenes/0018.png.K", Ks);*/




    cout << "Iniciando calculo de puntos SIFT..." << endl;
    vector <vector <KeyPoint > > keypoints;
    computeSift(imagenes, keypoints);

    vector <vector<DMatch> >matches;
    vector<Matchs> matchs;
    cout << "Iniciando calculo de puntos en correspondencia..." << endl;
    computeMatches(imagenes, keypoints, matches, matchs);

    Mat img_matches, F;

    cout << "Filtrando correspondencias..." << endl;
    for (unsigned int i = 0; i < matches.size(); i++) {

        F = matchesFilter(keypoints[matchs[i].kps1], keypoints[matchs[i].kps2], matches[matchs[i].matches]);
        if (matches[matchs[i].matches].size() < 60) {
            matchs[i].setInvalid();
        }
        else {
            matchs[i].setF(F);
        }
        cout << "Filtrado terminado(" << i+1 << "/" << matches.size()<< ")" << endl;
    }

    /*for (unsigned int i = 0; i < matchs.size(); i++) {
        if (matchs[i].isValid()) {
            drawMatches(imagenes[matchs[i].img1], keypoints[matchs[i].kps1], imagenes[matchs[i].img2], keypoints[matchs[i].kps2], matches[matchs[i].matches], img_matches);
            pintaI(img_matches);
        }
    }*/

    listaCorrespondencias list;
    cout << "Seleccionando correspondencias entre imagenes..." << endl;
    computeCorrespondences(matches, keypoints, matchs, list);

    /*for (unsigned int i = 0; i < list.size(); i++) {
        for (unsigned int j = 0; j < list[i].size(); j++) {
            cout << "Imagen: " << list[i][j].first << " " << "Punto(" << list[i][j].second.pt.x << ")(" << list[i][j].second.pt.y << ") ";
        }
        cout << endl;
    } */

    for (unsigned int i = 0; i < matchs.size(); i++) {
        if (matchs[i].isValid()) {
            evaluateMatch(imagenes[matchs[i].img1].size(), matches, keypoints, matchs[i]);
        }
    }

    sort(matchs.begin(), matchs.end(), compararMatchs);

    vector<Matx34d> Ps;
    vector<Point3d> pointcloud;
    vector<KeyPoint> correspImg1Pt;
    cout << "Iniciando triangulacion de las imagenes..." << endl;
    computeTriangulations(keypoints, matches, matchs, list, Ks, Ps, pointcloud, correspImg1Pt);

    cout << "Matrices de proyeccion calculadas: " << endl;
    for (unsigned int i = 0; i < Ps.size(); i++) {
        cout << Ps[i] << endl <<endl;
    }


    DepthIm(imagenes[matchs[0].img1].size(), pointcloud, correspImg1Pt);
    return 0;
}
