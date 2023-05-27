//
//  Matrix.cpp
//  NeuralNetwork
//
//  Created by Pham Tuan Binh on 18/03/2023.
//

#include "Matrix.hpp"
#define MIN -1
#define MAX 1

Matrix::Matrix(unsigned int rows, unsigned int cols, bool randomized)
{
    this->rows = rows;
    this->cols = cols;

    if (rows < 0 || cols < 0)
    {
        throw domain_error("Invalid dimensions");
    }

    allocateMemory();

    // initialize values for layer weights and nodes
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double val;
            if (randomized)
            {
                val = MIN + (double)rand() / (double)RAND_MAX * (MAX - MIN);
            }
            else
            {
                val = 0;
            }

            local_array[i][j] = val;
        }
    }
}

Matrix::Matrix(const Matrix &original)
{
    this->rows = original.rows;
    this->cols = original.cols;

    allocateMemory();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            local_array[i][j] = original.local_array[i][j];
        }
    }
}

Matrix::Matrix(ifstream &file)
{
    if (!file.is_open())
    {
        throw domain_error("Matrix can't be initialized from file.");
    }

    file >> rows >> cols;

    cout << rows << " " << cols << endl;

    allocateMemory();

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            file >> local_array[i][j];
        }
    }
}

Matrix::~Matrix()
{
    deallocateMemory();
}

Matrix &Matrix::operator=(const Matrix &original)
{

    if (this == &original)
    {
        return *this;
    }

    if (rows != original.rows || cols != original.cols)
    {
        deallocateMemory();

        rows = original.rows;
        cols = original.cols;
        allocateMemory();
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            local_array[i][j] = original.local_array[i][j];
        }
    }

    return *this;
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    if (rows != other.rows || cols != other.cols)
    {
        throw domain_error("Matrixes can't be added together: mismatch dimensions.");
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            local_array[i][j] += other.local_array[i][j];
        }
    }

    return *this;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    if (rows != other.rows || cols != other.cols)
    {
        throw domain_error("Matrixes can't be added together: mismatch dimensions.");
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            local_array[i][j] -= other.local_array[i][j];
        }
    }

    return *this;
}

Matrix &Matrix::operator*=(const Matrix &other)
{
    if (cols != other.rows)
    {
        throw domain_error("Matrixes can't be multiplied together: mismatch dimensions.");
    }

    Matrix temp(rows, other.cols, false);

    for (int i = 0; i < temp.rows; ++i)
    {
        for (int j = 0; j < temp.cols; ++j)
        {
            for (int k = 0; k < cols; ++k)
            {
                temp.local_array[i][j] += (local_array[i][k] * other.local_array[k][j]);
            }
        }
    }

    return (*this = temp);
}

Matrix &Matrix::operator*=(const double &a)
{
    Matrix temp(rows, cols, false);

    for (int i = 0; i < temp.rows; ++i)
    {
        for (int j = 0; j < temp.cols; ++j)
        {
            temp.local_array[i][j] = local_array[i][j] * a;
        }
    }

    return (*this = temp);
}

Matrix Matrix::transpose()
{

    Matrix temp(cols, rows, false);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            temp.local_array[j][i] = local_array[i][j];
        }
    }

    return temp;
}

// It is called hadamard function in Math, but I was stupid and sleep-deprived so dotMultiply it is.
Matrix Matrix::hadamard(const Matrix &other)
{
    if (cols != other.cols || rows != other.rows)
    {
        throw domain_error("Matrixes can't be dot multiplied together: mismatch dimensions.");
    }

    Matrix temp(rows, cols, false);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            temp.local_array[i][j] = local_array[i][j] * other.local_array[i][j];
        }
    }

    return temp;
}

void Matrix::allocateMemory()
{
    local_array = new double *[rows];
    for (int i = 0; i < rows; i++)
    {
        local_array[i] = new double[cols];
    }
}

void Matrix::deallocateMemory()
{
    for (int i = 0; i < rows; i++)
    {
        delete[] local_array[i];
    }

    delete[] local_array;
}

void Matrix::printMatrix()
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cout << local_array[i][j] << " ";
        }
        cout << endl;
    }
}

void Matrix::saveMatrix(ofstream &file)
{
    file << rows << " " << cols << endl;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            file << local_array[i][j] << " ";
        }
        file << endl;
    }
}

void Matrix::applyFunction(double (*func)(double))
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            local_array[i][j] = func(local_array[i][j]);
        }
    }
}

Matrix operator+(const Matrix &first, const Matrix &second)
{
    Matrix temporary(first);
    return (temporary += second);
}

Matrix operator-(const Matrix &first, const Matrix &second)
{
    Matrix temporary(first);
    return (temporary -= second);
}

Matrix operator*(const Matrix &first, const Matrix &second)
{
    Matrix temporary(first);
    return (temporary *= second);
}

Matrix operator*(const Matrix &first, const double &second)
{
    Matrix temporary(first);
    return (temporary *= second);
}
