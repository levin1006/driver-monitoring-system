// GVAPI.h
// Copyright General Vision Inc.
//
// Remark: use unsigned char instead of byte for compatibility with MatLab

#ifdef __cplusplus 
extern "C" {
#endif

#define DllExport __declspec( dllexport )

DllExport int Version();

//-------------------------------
//Cycle-accurate report functions
//-------------------------------
DllExport long ReadClockCounter();
DllExport void ClearClockCounter();
DllExport void EnableClockCounter();
DllExport void DisableClockCounter();

//-------------------------------
// Platform dependant functions
//-------------------------------

//Register Access Level functions
DllExport int Connect(int Platform, int DeviceID);
DllExport void Disconnect();
DllExport int Write_Addr(int addr, int length_inByte, unsigned char data[]);
DllExport int Read_Addr(int addr, int length_inByte, unsigned char data[]);
DllExport int Write(unsigned char module, unsigned char reg, int data);
DllExport int Read(unsigned char module, unsigned char reg);

//-------------------------------
// NeuroMem functions
//-------------------------------

//Loading and retrieval of neurons' content
DllExport int CountNeuronsandReset();
DllExport int CommittedNeurons();
DllExport int ClearNeurons();
DllExport int SaveNeurons(char *filename);
DllExport int LoadNeurons(char *filename);
DllExport int ReadNeurons(unsigned char *neurons, int ncount);
DllExport int WriteNeurons(unsigned char *neurons, int ncount);
DllExport void ReadNeuron(int neuronID, unsigned char model[], int* context, int* aif, int* minif, int* category);
DllExport int GetNeuronsInfo(int *MaxContext, int *MaxCategory);
DllExport int SurveyNeurons(unsigned char Context, int MaxCatValue, int CatHisto[], int CatDeg[]);

//Operations on single vector
DllExport int Learn(unsigned char vector[], int length, int category);
DllExport int BestMatch(unsigned char vector[], int length, int* distance, int* category, int* Nid);
DllExport int Recognize(unsigned char vector[], int length, int K, int distance[], int category[], int Nid[]);
DllExport void Broadcast(unsigned char vector[], int length);
DllExport void BroadcastSR(unsigned char vector[], int length);
DllExport void ReadLastVector(unsigned char vector[]);

//Operations on batches of vectors
DllExport int LearnVectors(unsigned char *vectors, int vectNbr, int vectLen, int *categories, bool ResetBeforeLearning, bool iterative);
DllExport void RecognizeVectors(unsigned char *vectors, int vectNbr, int vectLen, int K, int *distances, int *categories, int *nids);

DllExport int LearnIterative_wStats(unsigned char *vectors, int vectNbr, int vectLen, int *categories, bool ResetBeforeLearning, int *LearningCurve);
DllExport int BuildCodebook(unsigned char *vectors, int vectNbr, int vectLen, int CatAllocMode, int InitialCategory, bool WithoutUncertainty);
DllExport int BuildCodebookAdv(unsigned char *vectors, int vectNbr, int vectLen, int Maxif, long* error);
DllExport int Clusterize(unsigned char  *vectors, int vectNbr, int vectLen);
DllExport int MatchVectors(unsigned char *vectorsRef, int vectRefNbr, unsigned char *vectors, int vectNbr, int vectLen, int maxDistance, int *indexPairs, int *matchDistances);

//-------------------------------
// CogniSight functions
//-------------------------------

//CogniSight image manipulation functions
//DllExport void BufferPtrToCS(unsigned char  *imageBuffer, int Width, int Height, int BytePerPixel);

DllExport void BufferToCS(unsigned char  *imageBuffer, int Width, int Height, int BytePerPixel);
DllExport void GetCSBuffInfo(int *Width, int *Height, int *BytePerPixel);
DllExport void CSToBuffer(unsigned char  *imageBuffer);

//CogniSight ROI functions
DllExport void SetROI(int Width, int Height);
DllExport void GetROI(int *Width, int *Height);
DllExport void SetFeatID(int FeatID);
DllExport void GetFeatID(int *FeatID);
DllExport void SetFeatParams(int FeatID, int Normalize, int Minif, int Maxif, int Param1, int Param2);
DllExport void GetFeatParams(int *FeatID, int *Normalize, int *Minif, int *Maxif, int *Param1, int *Param2);
DllExport int GetFeature(int Left, int Top, unsigned char *Vector);
DllExport void SizeSubsample(int Width, int Height, int Monochrome, int KeepRatio);
DllExport int LearnROI(int Left, int Top, int Category);
DllExport int RecoROI(int Left, int Top, int *distance, int *category, int *nid);
DllExport int RecognizeROI(int Left, int Top, int K, int *distances, int *categories, int *nids);

//CogniSight ROS functions
DllExport void SetROS(int Left, int Top, int Width, int Height);
DllExport void GetROS(int *Left, int *Top, int *Width, int *Height);
DllExport int GetROSVectors(int stepX, int stepY, unsigned char  *Vectors, int *VLength);

DllExport int LearnROS(int stepX, int stepY, int category);
DllExport int BuildROSCodebook(int stepX, int stepY, int CatAllocMode);
DllExport int BuildROSCodebook_Fast(int stepX, int stepY, int *CtrX, int *CtrY);
DllExport int ROSToNeurons(int stepX, int stepY, int UsePositionAsContext);
DllExport void FindROSSalientBlocks(int stepX, int stepY, int K, int *CtrX, int *CtrY, int *AIF);
DllExport int FindROSUniqueBlocks(int stepX, int stepY, int K, int *CtrX, int *CtrY);

DllExport int FindROSObjects(int stepX, int stepY, int skipX, int skipY, int *CtrX, int *CtrY, int* distance, int* category, int* nid);
DllExport int FindROSAnomalies(int stepX, int stepY, int MaxNbr, int *CtrX, int *CtrY);
DllExport void MapCat(int stepX, int stepY, int *MaskMap, int *CatMap);
DllExport int MapROS(int stepX, int stepY, int *CatMap, int *DistMap, int *NidMap);
DllExport void GetObjectsBoundary(int *Left, int *Top, int *Width, int *Height);

//CogniSight Image functions
//DllExport int ImageToNeurons(int stepX, int stepY, int UsePositionAsContext);
//DllExport void FindSalientBlocks(int Nbr, int *CtrX, int *CtrY, int *AIF, int *Cat);

//CogniSight project functions
DllExport int SaveProject(char *filename);
DllExport int LoadProject(char *filename);

DllExport void ImageRGB2HSV(unsigned char *Hue, unsigned char *Saturation, unsigned char *Value);

//-------------------------------
// Text functions
//-------------------------------
#define MAX_LEN 1000; //(max length of input post, max length of output results, RAMbuffer size in NS4K reco controller
DllExport int BuildDictionnary(char *Filename, int NbrOfFrames, char *delimiter);
DllExport void SaveDictionnary(char *Filename);
DllExport int LoadDictionnary(char *Filename, int* wordCount, int* frameCount);
DllExport void SaveDictionnaryCSV(char *Filename);
DllExport int LoadDictionnaryCSV(char *Filename);
DllExport void ProcessPost(char* post, int* recognizedWords);
DllExport void BuildScores(int* recognizedWords, float* scoreTable);
DllExport void DisplayDictionnary();
DllExport void ReadDictionnary(char* words, float* weights);

// Copyright General Vision Inc.

#define MOD_TOP			0x52
#define MOD_CM1K		0x01
#define MOD_CS			0x08
#define MOD_CS0			0x06 //previous version used on V1KU rev 3 to 5 and .Net dll

//Description of the API
#define GV_PLATFORM	0x01
#define GV_VERSION	0x62

// Definition of the CM1K neuron registers
#define NM_NCR			0x00
#define NM_COMP			0x01
#define NM_LCOMP		0x02
#define NM_DIST			0x03 
#define NM_INDEXCOMP	0x03 
#define NM_CAT			0x04
#define NM_AIF			0x05
#define NM_MINIF		0x06
#define NM_MAXIF		0x07
#define NM_TESTCOMP		0x08
#define NM_TESTCAT		0x09
#define NM_NID			0x0A
#define NM_GCR			0x0B
#define NM_RESETCHAIN	0x0C
#define NM_NSR			0x0D
#define NM_NCOUNT		0x0F	
#define NM_FORGET		0x0F

//CM1K reco logic registers
#define NM_LEFT			0x11
#define NM_TOP			0x12
#define NM_NWIDTH		0x13
#define NM_NHEIGHT		0x14
#define NM_BWIDTH		0x15
#define NM_BHEIGHT		0x16
#define NM_RSR			0x1C
#define NM_RTDIST		0x1D
#define NM_RTCAT		0x1E
#define NM_ROIINIT		0x1F

// Definition of default values
#define MAXVECLENGTH	256

//CogniSight registers (module 8-9)
#define CS_WIDTH		0x81
#define CS_HEIGHT		0x82
#define CS_FEATID		0x83
#define CS_FEATNORMALIZE	0x84
#define CS_FEATMINIF	0x85
#define CS_FEATMAXIF	0x86
#define CS_FEATPARAM1	0x87
#define CS_FEATPARAM2	0x88
#define CS_ROSLEFT		0x89
#define CS_ROSTOP		0x8A
#define CS_ROSWIDTH		0x8B
#define CS_ROSHEIGHT	0x8C
#define CS_STEPX		0x8D
#define CS_STEPY		0x8E
#define CS_SKIPX		0x8F
#define CS_SKIPY		0x90

// equivalence with the V1KU registers and .Net API
#define CS0_ROSLEFT		0x66
#define CS0_ROSTOP		0x67
#define CS0_ROSWIDTH	0x68
#define CS0_ROSHEIGHT	0x69
#define CS0_STEPX		0x73
#define CS0_STEPY		0x74
#define CS0_FEATID		0x80
//#define CS_LEFT			0x61
//#define CS_TOP			0x62
//#define CS_CSR			0x60
//#define CS_RECODIST		0x63
//#define CS_RECOCAT		0x64
//#define CS_CATL			0x65
//#define CS_HITCOUNT		0x6A
//#define CS_HITX			0x6B
//#define CS_HITY			0x6C
//#define CS_HITDIST		0x6D
//#define CS_HITCAT		0x6E
//#define CS_RSR			0x75

#ifdef __cplusplus 
}
#endif
