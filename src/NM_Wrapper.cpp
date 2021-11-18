// 2017.09.01 
///          Separation from main file


#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "opencv2/highgui.hpp"
#include "GVAPI.h"
#include "DMS_Algo_Global_Parameter.h"
#include "NM_Wrapper.h"

static char buf_char[256];


// Check NM connection and the number of available neurons
void NM_Initialization()
{
	if( Connect(0,0) )
	{
		printf( "[ERROR] Could not connect...\n" );
		system( "pause" );
	}


	sprintf_s( buf_char, "[SYSTEM] Current version: %d\n", Version() );
	printf( buf_char );

	int maxNeurons = CountNeuronsandReset();
	if( maxNeurons != _NEURON_MAX_AVAILABLE )
	{
		printf( "[ERROR] The number of available neurons is not matched with definition." );
		system( "pause" );
	}
	else
	{
		sprintf_s( buf_char, "[SYSTEM] Neurons Available = %d\n", maxNeurons );
		printf( buf_char );
		ClearNeurons();
	}
	
	//Write( 1, NM_MINIF, _NM_LEARNING_INIT_MINIF );
	//Write( 1, NM_MAXIF, _NM_LEARNING_INIT_MAXIF );
}

int NM_ReadKnowledge( int ncount, unsigned char *neurons )
{	
	/// Read neuron information
	ncount = ReadNeurons( neurons, ncount );

#if _OUTPUT_DISP_LOG_NEURONS_STATE
		for( int idx = 0 ; idx < ncount ; idx++ )
		{
			int AifIdx = ( idx * _NEURON_CONTENT_SIZE ) + _NEURON_CONTENT_AIF;
			int MifIdx = ( idx * _NEURON_CONTENT_SIZE ) + _NEURON_CONTENT_MIF;
			int CatIdx = ( idx * _NEURON_CONTENT_SIZE ) + _NEURON_CONTENT_CAT;
			int AIF = ( neurons[ AifIdx ] << 8 ) + neurons[ AifIdx + 1 ];
			int MIF = ( neurons[ MifIdx ] << 8 ) + neurons[ MifIdx + 1 ];
			int CAT = ( neurons[ CatIdx ] << 8 ) + neurons[ CatIdx + 1 ];
			sprintf_s( buf_char, "- Neuron #%02u: CAT = %u, AIF = %u, MIF = %u\n", idx + 1, CAT, AIF, MIF );
			disp( buf_char );
		}
#endif

	return ncount;
}

// Load and Read knowledge
int NM_LoadKnowledge( char *fileName )
{
	int ncount;

	/// Load knowledge file
	printf( "[SYSTEM] Loading neurons from knowledge file\n" );

	ncount = LoadNeurons( fileName );
	sprintf_s( buf_char, "[INFO] #Commintted neurons = %d\n", ncount );
	printf( buf_char );

#if _OUTPUT_DISP_LOG_NEURONS_STATE
	unsigned char neurons[_NEURON_MAX_AVAILABLE * _NEURON_CONTENT_SIZE];
	NM_ReadKnowledge( ncount, neurons );
#endif

	return ncount;
}



