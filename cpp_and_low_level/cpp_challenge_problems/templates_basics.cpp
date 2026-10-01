// 9/25/2026

template <typename T>
T my_max(const T &x, const T &y)
{
    return (x > y) ? x : y;
}

template <typename X>
class pair
{
private:
    X a;
    X b;

public:
    pair(const X &a, const X &b) : a(a), b(b) {}

    X get_a() { return a; }
    X get_b() { return b; }

    X sum() { return a + b; }
};