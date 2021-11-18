#pragma once

// General definition
#define _EPS									0.000001		// 1u

#define _TYPE_SHORT_MAX							16383

#define TRUE									1
#define FALSE									0

#define _CHAR_SLASH								"/"
#define _EXTENSION_KNF							".knf"

// NM related parameter definition
#define _NEURON_MAX_AVAILABLE					1024
#define _NEURON_MAX_VECTOR_SIZE					256
#define _NEURON_CONTENT_SIZE					264
#define _NEURON_CONTENT_NCR						1
#define _NEURON_CONTENT_VECTOR					2	// Vector size = 256 bytes
#define _NEURON_CONTENT_AIF						258	// Memory size = 2 bytes
#define _NEURON_CONTENT_MIF						260	// Memory size = 2 bytes
#define _NEURON_CONTENT_CAT						262	// Memory size = 2 bytes


#define _EYE_CLOSED								0
#define _EYE_OPENED								1

#define _EYE_GAZE_CENTER						0
#define _EYE_GAZE_LEFT							1
#define _EYE_GAZE_RIGHT							2
#define _EYE_GAZE_NONE							3

#define _NM_EYE_OPENED							0
#define _NM_EYE_CLOSED							1

#define _KNF_FACE								1
#define _KNF_EYE_CLOSED							2


#define _INPUT_IMAGE_WIDTH						640
#define _INPUT_IMAGE_HEIGHT						480
#define	_INPUT_FRAME_RATE						30
