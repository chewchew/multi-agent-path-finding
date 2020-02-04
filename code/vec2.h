#if !defined(VEC2_H)
#define VEC2_H

struct Vec2
{
    f32 x;
    f32 y;
};
    
static Vec2
operator-(Vec2 v1, Vec2 v2)
{
    return {v1.x - v2.x, v1.y - v2.y};
}

static Vec2
operator+(Vec2 v1, Vec2 v2)
{
    return {v1.x + v2.x, v1.y + v2.y};
}

static Vec2
operator+(Vec2 v1, f32 x)
{
    return {v1.x + x, v1.y + x};
}

static void
operator+=(Vec2& v1, Vec2 v2)
{
    v1 = v1 + v2;
}

static Vec2
operator*(Vec2 v, f32 x)
{
    return {v.x * x, v.y * x};
}

static f32
dot(Vec2 v1, Vec2 v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

static f32
distance(Vec2 v1, Vec2 v2)
{
    f32 x = v1.x - v2.x;
    f32 y = v1.y - v2.y;
    return sqrtf(x * x + y * y);
}

#endif
