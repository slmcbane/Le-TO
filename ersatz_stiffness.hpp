#ifndef ERSATZ_STIFFNESS_HPP
#define ERSATZ_STIFFNESS_HPP

#include <cmath>
#include <cstdio>
#include <exception>

struct InterpException : public std::exception
{
};

/*
 * Used to interpolate the stiffness of a material with intermediate density
 * according to s(rho) = (epsilon + (1-epsilon)*rho)^p
 *
 * This construction ensures that if p < 1, the derivative w.r.t. rho is not
 * infinite, so that the lower bound on rho in an optimization can be 0.
 */
struct ErsatzStiffness
{
    double p;
    double epsilon; // << 1. E_min = epsilon^p E_0.

    double operator()(double rho) const noexcept
    {
        const double adjusted = epsilon + (1 - epsilon) * rho;
        return std::pow(adjusted, p);
    }

    double derivative(double rho) const noexcept
    {
        const double adjusted = epsilon + (1 - epsilon) * rho;
        return p * std::pow(adjusted, p - 1) * (1 - epsilon);
    }

    constexpr ErsatzStiffness(double power, double eps) noexcept : p{power}, epsilon{eps} {}

    void serialize(FILE *out) const
    {
        int count = fwrite(&p, sizeof(p), 1, out);
        if (count != 1)
        {
            throw InterpException();
        }
        count = fwrite(&epsilon, sizeof(double), 1, out);
        if (count != 1)
        {
            throw InterpException();
        }
    }

    static ErsatzStiffness deserialize(FILE *in)
    {
        double p, eps;
        int count = fread(&p, sizeof(double), 1, in);
        if (count != 1)
        {
            throw InterpException();
        }
        count = fread(&eps, sizeof(double), 1, in);
        if (count != 1)
        {
            throw InterpException();
        }
        return ErsatzStiffness(p, eps);
    }
};

#endif // SIMP_HPP
