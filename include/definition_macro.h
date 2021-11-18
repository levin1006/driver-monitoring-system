#pragma once


#define CLAMP( x, a, b )			( ( x > b ) ? b : ( ( x < a) ? a : x ) )

#define SQUARE( x )					( x * x )

#define ROUND( x )					( ( x < 0 ) ? ceil( x - 0.5 ) : floor( x + 0.5 ) )

#define ISINSIDE( x, a, b )			( ( x >= a ) && ( x < b ) )

inline unsigned char min_( unsigned char x, unsigned char y )
{
	return ( x <= y ) ? x : y;
}

inline int min_( int x, int y )
{
	return ( x <= y ) ? x : y;
}

inline float min_( float x, float y )
{
	return ( x <= y ) ? x : y;
}

inline unsigned char max_( unsigned char x, unsigned char y )
{
	return ( x >= y ) ? x : y;
}

inline int max_( int x, int y )
{
	return ( x >= y ) ? x : y;
}

inline float max_( float x, float y )
{
	return ( x >= y ) ? x : y;
}

inline int clamp_( int x, int a, int b )
{
	return ( x > b ) ? b : ( ( x < a) ? a : x );
}

inline float sq_( float x )
{
	return x * x;
}

inline int sq_( int x )
{
	return x * x;
}

inline int idx_( int x, int y, int widthStep )
{
	return x + y * widthStep;
}

inline int isinside_( int x, int a, int b )
{
	return ( x >= a ) && ( x < b );
}
