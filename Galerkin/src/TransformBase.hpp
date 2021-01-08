#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include "utils.hpp"

#include <exception>

namespace Galerkin
{

/*!
 * @brief Namespace for functionality related to coordinate transformations.
 */
namespace Transforms
{

template <auto N, class Derived>
class TransformBase
{
public:
    /*!
     * @brief Call operator just forward arguments to the derived operator.
     */
    template <class... T>
    constexpr auto operator()(const T&... args) const noexcept
    {
        return derived()(args...);
    }

    /*!
     * @brief Compute partial derivatives using change of variables rule.
     * 
     * Returns the derivative of a function `g` defined by the relation
     * `g(x) = f(inverse_transform(x))`; `f` is the argument given to `partial`.
     * The template argument is the index of the variable to differentiate with
     * respect to. The returned object is a function that returns the derivative
     * of `g` with respect to `x`, as a function of `xi` where `xi` is the inverse
     * transform of `x`.
     * 
     * For example if `f(x) = x^2`, and the transform is a uniform scaling of the
     * interval (-1, 1) to (-2, 2), then `transform.partial<0>(f)` is equivalent
     * to a function `df/dx(x) = x`, since `d(xi)/dx == 1/2`.
     * 
     * Requires that `f` satisfy the interface imposed by `FunctionBase`.
     * 
     * @see FunctionBase.hpp
     */
    template <auto I, class F>
    constexpr auto partial(const F& f) const noexcept
    {
        auto first_term = Galerkin::partial<0>(f) * derived().template inv_jacobian<0, I>();
        return static_sum<1, N>(
            [&](auto J) { return Galerkin::partial<J()>(f) * derived().template inv_jacobian<J(), I>(); },
            first_term
        );
    }

    /*!
     * @brief Integrate `f` over the transformed region.
     * 
     * The template parameter (>= 0) specifies a desired order of accuracy of the
     * integration; polynomials of order less than or equal to `I` will be
     * integrated exactly, assuming that the derived class faithfully implements
     * `quadrature`.
     * 
     * The function integrated is `g(x) = f(inverse_transform(x))`, where `x` is
     * the transformed coordinate. The argument to the function is `f` in the
     * above equation.
     */
    template <auto I, class F>
    constexpr auto integrate(const F& f) const noexcept
    {
        return derived().template quadrature<I>(f * derived().detJ());
    }
    
private:
    TransformBase() = default;
    friend Derived;

    constexpr const auto& derived() const noexcept { return static_cast<const Derived&>(*this); }
};

/// Used as a tag type to indicate that a transform's constructor should check geometric compatibility.
struct GeometryCheck {};

/// Thrown for failed geometry checks.
struct GeometryException : public std::exception {};

} /* namespace Transforms */

} /* namespace Galerkin */

#endif /* TRANSFORMS_HPP */
