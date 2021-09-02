#ifndef SAVE_EIGEN_HPP
#define SAVE_EIGEN_HPP

#include <Eigen/Core>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

inline void save_eigen(const Eigen::VectorXd &x, const char *name)
{
    std::FILE *ofile = fopen(name, "wb");
    if (!ofile)
    {
        fprintf(stderr, "Failed to open output file %s\n", name);
        exit(1);
    }
    int64_t sz = x.rows();

    fwrite(&sz, sizeof(sz), 1, ofile);
    fwrite(x.data(), sizeof(double), x.rows(), ofile);
    fclose(ofile);
}

inline Eigen::VectorXd read_eigen(const char *name)
{
    std::FILE *infile = fopen(name, "rb");
    if (!infile)
    {
        fprintf(stderr, "Failed to open input file %s\n", name);
        exit(1);
    }
    int64_t sz;

    int count = fread(&sz, sizeof(int64_t), 1, infile);
    if (count != 1)
    {
        fprintf(stderr, "Unkown error in fread\n");
        exit(1);
    }

    Eigen::VectorXd result(sz);
    count = fread(result.data(), sizeof(double), sz, infile);
    if (count != sz)
    {
        fprintf(stderr, "Unknown error in fread\n");
        exit(1);
    }
    fclose(infile);

    return result;
}

#endif // SAVE_EIGEN_HPP
