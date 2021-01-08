/*
 * Copyright (c) 2019, Sean McBane and The University of Texas at Austin.
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

/*!
 * @file utils.hpp
 * @brief Supporting utilities for the Galerkin library.
 * @author Sean McBane <sean.mcbane@protonmail.com>
 */

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

/*!
 * @brief All library functionality is contained in the Galerkin namespace.
 */
namespace Galerkin
{

namespace
{
    template <class T, int N>
    struct ntuple_impl
    {
        typedef decltype(
            std::tuple_cat(
                std::make_tuple(std::declval<T>()),
                std::declval<typename ntuple_impl<T, N-1>::type>()
            )
        ) type;
    };

    template<class T>
    struct ntuple_impl<T, 0>
    {
        typedef std::tuple<> type;
    };
}

/// ntuple is a convenient alias for a tuple of N elements of the same type.
template <class T, int N>
using ntuple = typename ntuple_impl<T, N>::type;

/// Template constant yields 1 cast to the given type.
template <class T>
constexpr T one = static_cast<T>(1);

/// Template constant yields 0 cast to the given type.
template <class T>
constexpr T zero = static_cast<T>(0);

namespace
{
    template <auto... Ns, class... Ts>
    constexpr auto tuple_tail_impl(std::index_sequence<Ns...>, std::tuple<Ts...> t) noexcept
    {
        return std::tuple(std::get<Ns+1>(t)...);
    }

    template <auto... Ns, class T>
    constexpr auto tuple_tail_impl(std::index_sequence<Ns...>, const std::array<T, sizeof...(Ns)+1> &a) noexcept
    {
        return std::array<T, sizeof...(Ns)>{ std::get<Ns+1>(a)... };
    }
}

template <class T, class... Ts>
constexpr auto tuple_tail(std::tuple<T, Ts...> t) noexcept
{
    return tuple_tail_impl(std::index_sequence_for<Ts...>(), t);
}

template <class T, auto N>
constexpr auto tuple_tail(const std::array<T, N> &a) noexcept
{
    static_assert(N > 0, "Can't take tail for 0-element array");
    return tuple_tail_impl(std::make_index_sequence<N-1>(), a);
}

/*!
 * @brief Do an arbitrary reduction on possibly compile-time expressions.
 * 
 * This performs the reduction
 * 
 *     auto y = x;
 *     for (auto i = BEGIN; i < END; i += STEP)
 *     {
 *         y = c(y, f(i));
 *     }
 *     return y;
 * 
 * The above loop must be "type-stable", but this facility performs the operation
 * in a recursive manner so the type of `y` might change every loop iteration.
 * In addition, `f` receives the "loop index" as an `integral_constant` so that
 * the function argument can be used to specify a template parameter (e.g. `std::get`).
 * 
 * Useful for doing sums and products with compile-time types like `Rational` and
 * `Multinomial`.
 * 
 * @tparam BEGIN The initial value for the reduction loop
 * @tparam END The end value for the reduction loop; range is `[BEGIN, END)`.
 * @tparam STEP The increment between evaluations.
 * @param[in] f A generic callable object that accepts a `std::integral_constant`.
 * @param[in] x An initial value for the reduction - e.g. `zero<T>` for a sum or
 * `one<T>` for a product.
 * @param[in] c A callable taking two objects `x` and `y` and returning a single
 * value; for example, `operator+` to perform a summation.
 * 
 * @returns The reduction using `x` as an initial value.
 */
template <auto BEGIN, auto END, auto STEP, class F, class COMB, class T>
constexpr auto static_reduce(F&& f, T x, COMB&& c)
{
    if constexpr (BEGIN >= END)
    {
        return x;
    }
    else
    {
        auto y = std::forward<F>(f)(std::integral_constant<decltype(BEGIN), BEGIN>());
        return static_reduce<BEGIN+STEP, END, STEP>(
            std::forward<F>(f), std::forward<COMB>(c)(x, y), std::forward<COMB>(c)
        );
    }
}

template <auto BEGIN, auto END, auto STEP, class F>
constexpr void static_for(F&& f)
{
    if constexpr (BEGIN >= END)
    {
        return;
    }
    else
    {
        std::forward<F>(f)(std::integral_constant<decltype(BEGIN), BEGIN>());
        static_for<BEGIN+STEP, END, STEP>(std::forward<F>(f));
    }
}

/*!
 * @brief A simple wrapper around `static_reduce` that performs a summation.
 * 
 * This wrapper also provides a default value for the `STEP` parameter; as a
 * consequence if you want to override it you will have to specify types of
 * `f` and `x` in the template parameters.
 * 
 * @param[in] f A callable taking an `integral_constant` as for `static_reduce`.
 * @param[in] x Initial value for the summation.
 * 
 * @returns `x + sum(f(i) for i in range(BEGIN, STEP, END)`
 */
template <auto BEGIN, auto END, class F, class T, auto STEP=1>
constexpr auto static_sum(F&& f, T x)
{
    constexpr auto comb = [](auto x, auto y) { return x + y; };
    return static_reduce<BEGIN, END, STEP>(std::forward<F>(f), x, comb);
}

/*!
 * @brief Holds a list of types, which should represent values and be default-constructible.
 * 
 * The metaprogramming in Galerkin uses a paradigm where mathematical entities are
 * represented via types that hold all of their parameters as template parameters;
 * however, these objects are passed around *by value*, not only as template
 * parameters. This allows for mixing compile-time computations with run-time all
 * in the same code. The `typeconst_list` is my list type for holding values that
 * act like this. It expects `Types...` to be default-constructible and, for some
 * methods, comparable using `<=` and `==`.
 */
template <class... Types>
struct typeconst_list;

/*!
 * @brief Functional-style construction of a `typeconst_list`.
 */
template <class... Types>
constexpr auto make_list(Types...)
{
    return typeconst_list<Types...>();
}

/*!
 * @brief Specialization of `typeconst_list` for the empty list.
 * 
 * Accessors like `head` and `tail` are undefined for this base type; other
 * methods may throw a `static_assert` if used here. Permitted operations should
 * be common sense.
 */
template<>
struct typeconst_list<>
{
    /// Sort the list; trivially for the empty list.
    static constexpr auto sorted()
    {
        return typeconst_list<>();
    }

    /// Append another list to this one - again trivial.
    template <class... Types>
    static constexpr auto append(typeconst_list<Types...> lst)
    {
        return lst;
    }

    /// Retrieve the number of items in the list.
    static constexpr auto count = 0UL;
    static constexpr size_t size() noexcept { return count; }

    /// Base specialization of `lst.map`; a no-op for the empty specialization.
    template <class F>
    static constexpr auto map([[maybe_unused]] F&& f) { return typeconst_list<>(); }

    /// Get only unique members of the list.
    static constexpr auto unique() { return typeconst_list<>(); }
};

/// Specialization of `typeconst_list` for the case where the list is *not* empty.
template <class T, class... Types>
struct typeconst_list<T, Types...>
{
    /// Return another list with `Types...` order with `<=`. `Types...` must be
    /// default constructible and have `<=` defined.
    static constexpr auto sorted()
    {
        // base case, only one type in the list.
        if constexpr (sizeof...(Types) == 0)
        {
            return typeconst_list<T>();
        }
        else
        {
            constexpr auto sorted_tail = tail().sorted();
            if constexpr(head() <= sorted_tail.head())
            {
                return typeconst_list<T>().append(sorted_tail);
            }
            else
            {
                constexpr auto first_two = make_list(sorted_tail.head(), T());
                return first_two.append(sorted_tail.tail()).sorted();
            }
        }
    }

    /*!
     * @brief Get a list containing only the unique types from this one.
     * 
     * This method expects the list to be sorted - lst.sorted().unique() does
     * what you want if the list is not sorted, except for preserve ordering.
     */
    static constexpr auto unique()
    {
        if constexpr (sizeof...(Types) == 0)
        {
            return typeconst_list<T>();
        }
        else
        {
            if constexpr (T() == tail().head())
            {
                return tail().unique();
            }
            else
            {
                return make_list(T()).append(tail().unique());
            }
        }
    }

    /// Append another list to this one, returning the new list.
    template <class... OtherTypes>
    static constexpr auto append(typeconst_list<OtherTypes...>)
    {
        return typeconst_list<T, Types..., OtherTypes...>();
    }

    /// Retrieve the first element in the list.
    static constexpr auto head() { return T(); }

    /// Retrieve all the elements of the list but the first.
    static constexpr auto tail() { return typeconst_list<Types...>(); }

    /// Get the number of elements in the list.
    static constexpr auto count = 1 + sizeof...(Types);
    static constexpr auto size() noexcept { return count; }

    /// Return a list containing the types returned by applying `f` elementwise
    /// to default-constructed instances of `Types...`.
    template <class F>
    static constexpr auto map(F&& f)
    {
        return make_list(std::forward<F>(f)(T()), std::forward<F>(f)(Types())...);
    }
};

template <auto v>
constexpr auto intgr_constant = std::integral_constant<decltype(v), v>();

template <class T>
constexpr bool is_intgr_constant = false;

template <auto v>
constexpr bool is_intgr_constant<std::integral_constant<decltype(v), v>> = true;

template <class Lower, class Upper, auto N>
constexpr auto evenly_spaced(Lower, Upper, std::integral_constant<decltype(N), N>) noexcept
{
    constexpr auto delta = (Upper{} - Lower{}) / intgr_constant<N>;
    return static_reduce<0, N+1, 1>(
        [=](auto I) { return Lower{} + I * delta; },
        typeconst_list<>{},
        [](auto lst, auto x) { return lst.append(make_list(x)); }
    );
}

/// Get the I-th element (0-based) from the list given.
template <auto I, class... Types>
constexpr auto get(typeconst_list<Types...> lst)
{
    static_assert(I >= 0 && I < lst.count, "Out of bounds access to list");
    if constexpr (I == 0)
    {
        return lst.head();
    }
    else
    {
        return get<I-1>(lst.tail());
    }
}

/// Test list equality; used in tests primarily.
template <class... T1, class... T2>
constexpr bool operator==(typeconst_list<T1...>, typeconst_list<T2...>) noexcept
{
    return std::is_same_v<typeconst_list<T1...>, typeconst_list<T2...>>;
}

/// Repeat an item N times to create a list.
template <auto N, class T>
constexpr auto repeatedly(T) noexcept
{
    if constexpr (N == 0)
    {
        return typeconst_list<>();
    }
    else
    {
        return typeconst_list<T>().append(repeatedly<N-1>(T()));
    }
}

template <auto I, class T>
constexpr auto partial(const T& x) noexcept
{
    return x.template partial<I>();
}

} /* namespace Galerkin */

#endif /* UTILS_HPP */
