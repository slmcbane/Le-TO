/*
 * Copyright (c) 2019, Sean McBane and The University of Texas at Austin.
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef FUNCTIONBASE_HPP
#define FUNCTIONBASE_HPP

/*!
 * @file FunctionBase.hpp
 * @brief Functionality (ha) related to functions from `R^n -> R`.
 */

#include "utils.hpp"
#include "Rationals.hpp"
#include "Metanomials.hpp"

#include <cmath>

namespace Galerkin
{

/*!
 * @brief Functions contains functionality for **scalar** functions
 * 
 * The utilities in this namespace exist to do calculus and arithmetic with
 * functions from R^n -> R. It contains a base class that serves to provide
 * fallback methods for summation and products of functions, and to compute their
 * partial derivatives.
 */
namespace Functions
{

template <class Derived>
class FunctionBase;

/*!
 * @brief Represents the sum of two scalar functions.
 * 
 * Given two functions `f` and `g`, `FunctionSum(f, g)` returns a new scalar
 * function that when called, returns `f(x) + g(x)` and whose partial derivative
 * with respect to `x_i` is the sum of the partial derivatives of `f` and `g`.
 */
template <class F1, class F2>
class FunctionSum : public FunctionBase<FunctionSum<F1, F2>>
{
public:
    constexpr FunctionSum(const F1 &a, const F2 &b) : m_f1(a), m_f2(b) {}

    template <auto I>
    constexpr auto partial() const noexcept
    {
        return Galerkin::partial<I>(m_f1) + Galerkin::partial<I>(m_f2);
    }

    template <class... T>
    constexpr auto operator()(const T &... x) const noexcept
    {
        return m_f1(x...) + m_f2(x...);
    }

private:
    F1 m_f1;
    F2 m_f2;
};

/*!
 * @brief Represent the product of two scalar functions
 * 
 * Given two functions `f` and `g`, `FunctionProduct(f, g)(x)` evaluates
 * `f(x) * g(x)`, and has the appropriate partial derivatives as well.
 */
template <class F1, class F2>
class FunctionProduct : public FunctionBase<FunctionProduct<F1, F2>>
{
public:
    constexpr FunctionProduct(const F1 &a, const F2 &b) : m_f1(a), m_f2(b) {}

    template <auto I>
    constexpr auto partial() const noexcept
    {
        return Galerkin::partial<I>(m_f1) * m_f2 + m_f1 * Galerkin::partial<I>(m_f2);
    }

    template <class... T>
    constexpr auto operator()(const T &... x) const noexcept
    {
        return m_f1(x...) * m_f2(x...);
    }

private:
    F1 m_f1;
    F2 m_f2;
};

/*!
 * @brief Given a function `f`, represents the function `-f`.
 */
template <class F>
class FunctionNegation : public FunctionBase<FunctionNegation<F>>
{
public:
    constexpr FunctionNegation(const F& f) : m_f(f) {}

    template <auto I>
    constexpr auto partial() const noexcept
    {
        return FunctionNegation(Galerkin::partial<I>(m_f));
    }

    template <class... T>
    constexpr auto operator()(const T&... x) const noexcept
    {
        return -m_f(x...);
    }

private:
    F m_f;
};

/*!
 * @brief Represent the quotient of two scalar functions
 * 
 * Given scalar functions `f` and `g`, `FunctionQuotient(f, g)(x)` evaluates
 * `f(x) / g(x)` and also has the appropriate partial derivatives.
 */
template <class F1, class F2>
class FunctionQuotient : public FunctionBase<FunctionQuotient<F1, F2>>
{
public:
    constexpr FunctionQuotient(const F1 &a, const F2 &b) : m_f1(a), m_f2(b) {}

    template <auto I>
    constexpr auto partial() const noexcept
    {
        return (m_f2 * Galerkin::partial<I>(m_f1) -
            m_f1 * Galerkin::partial<I>(m_f2)) / (m_f2 * m_f2);
    }

    template <class... T>
    constexpr auto operator()(const T &... x) const noexcept
    {
        return m_f1(x...) / m_f2(x...);
    }

private:
    F1 m_f1;
    F2 m_f2;
};

/*!
 * @brief CRTP base class providing functionality for scalar functions.
 * 
 * Classes that derive from `FunctionBase` inherit a suite of operator
 * functionality; they may now be used in expressions like `(f*g)(x)` or
 * `(f+g)*h`. This is accomplished using the derived classes `FunctionSum`,
 * `FunctionProduct`, `FunctionQuotient`, and `FunctionNegation`. The derived
 * class is expected to implement a call operator, and a member function
 * template `partial<I>()` that returns a new function object that when called,
 * evaluates the partial derivative with respect to the `I`-th component of the
 * function's vector argument (functions are `f: R^n -> R`).
 * 
 * The one piece of functionality not implemented is composition of functions
 * and the chain rule; I haven't needed it yet.
 */
template <class Derived>
class FunctionBase
{
public:
    template <class Other>
    constexpr auto operator+(const Other &other) const noexcept
    {
        return FunctionSum(static_cast<const Derived&>(*this), other);
    }

    template <class Other>
    constexpr auto operator-(const Other &other) const noexcept
    {
        return *this + FunctionNegation(other);
    }

    constexpr auto operator-() const noexcept
    {
        return FunctionNegation(static_cast<const Derived&>(*this));
    }

    template <class Other>
    constexpr auto operator*(const Other &other) const noexcept
    {
        return FunctionProduct(static_cast<const Derived&>(*this), other);
    }

    template <class Other>
    constexpr auto operator/(const Other &other) const noexcept
    {
        return FunctionQuotient(static_cast<const Derived&>(*this), other);
    }

    template <class... T>
    constexpr auto operator()(const T &... x) const noexcept
    {
        return static_cast<const Derived*>(*this)->operator()(x...);
    }

private:
    FunctionBase() = default;
    friend Derived;
};

/*!
 * @brief Represents the scalar function `f(x) = c` with `c` a constant.
 * 
 * Construct a `ConstantFunction` returning the value `x` by `ConstantFunction(x)`;
 * the new object, when called, returns `x` and its partial derivatives return
 * `ConstantFunction(Rationals::rational<0>)`.
 */
template <class T>
class ConstantFunction : public FunctionBase<ConstantFunction<T>>
{
public:
    constexpr ConstantFunction(T v) noexcept : m_val(v) {}

    template <class... Args>
    constexpr auto operator()([[maybe_unused]] const Args&... args) const noexcept
    {
        return m_val;
    }

    template <auto I>
    constexpr auto partial() const noexcept
    {
        return ConstantFunction(Rationals::rational<0>);
    }
private:
    T m_val;
};

/********************************************************************************
 * Tests for ConstantFunction
 *******************************************************************************/
#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::Functions] Test ConstantFunction")
{
    constexpr auto fn = ConstantFunction(2.0);
    REQUIRE(fn() == 2.0);
    REQUIRE(fn(1, 2, 3) == 2.0);
    REQUIRE(fn(std::tuple('a', 'b', 'c')) == 2.0);

    REQUIRE(partial<0>(fn)(1) == Rationals::rational<0>);
    REQUIRE(partial<10>(fn)(std::tuple(0, 0)) == Rationals::rational<0>);
}

#endif // DOCTEST_LIBRARY_INCLUDED
/********************************************************************************
 * End test block.
 *******************************************************************************/

/*!
 * @brief Represents the raising of a function to a power.
 * 
 * The type T of the exponent is a Rationals::rational; thus it can be stored
 * nowhere.
 */
template <class F, class T>
class PowerFunction : public FunctionBase<PowerFunction<F, T>>
{
public:
    constexpr PowerFunction(const F &f, [[maybe_unused]] T pow) : m_fn(f)
    {
    }

    template <class... Args>
    constexpr auto operator()(const Args &... args) const
    {
        return std::pow(m_fn(args...), static_cast<double>(T{}));
    }

    template <auto I>
    constexpr auto partial() const noexcept
    {
        constexpr auto new_power = T{} - Rationals::rational<1>;
        const auto inner_partial = m_fn.template partial<I>();
        return PowerFunction<F, std::decay_t<decltype(new_power)>>(m_fn, new_power)
            * ConstantFunction(T{}) * inner_partial;
    }

private:
    F m_fn;
};

} // namespace Functions

} // namespace Galerkin

#endif /* FUNCTIONBASE_HPP */