//
//  Matrix.hpp
//  NeuralNetwork
//
//  Created by Pham Tuan Binh on 18/03/2023.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <time.h>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace std;

// This matrix is specifically built to accommodate layers weights calculation, so certain functions for Matrix isn't written.
class Matrix
{
public:
    unsigned int rows, cols;
    double **local_array;

    Matrix(unsigned int rows = 1, unsigned int cols = 1, bool randomized = true);

    Matrix(const Matrix &original);

    Matrix(ifstream &file);

    ~Matrix();

    Matrix &operator=(const Matrix &original);

    Matrix &operator+=(const Matrix &other);

    Matrix &operator-=(const Matrix &other);

    Matrix &operator*=(const Matrix &other);

    Matrix &operator*=(const double &a);

    Matrix transpose();

    Matrix hadamard(const Matrix &other);

    void applyFunction(double (*func)(double));

    void allocateMemory();

    void deallocateMemory();

    void printMatrix();

    void saveMatrix(ofstream &file);

    // Normally we have to have to overload member functions like the above but it made it more complicated so I stick with the non-member function
    friend Matrix operator+(const Matrix &first, const Matrix &second);
    friend Matrix operator*(const Matrix &first, const Matrix &second);
    friend Matrix operator*(const Matrix &first, const double &second);
    friend Matrix operator-(const Matrix &first, const Matrix &second);
};

#endif /* Matrix_hpp */
