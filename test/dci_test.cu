/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dci.cu
 * Author: e.vicari
 *
 * Created on 1 marzo 2016, 12.12
 */

#include <cstdlib>
#include "dci.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    
    // ********************************************************************* //
    // INITIALIZATION SECTION - call only once, store app object globally    //
    // ********************************************************************* //
    
    // create default configuration
    dci::RunInfo configuration = dci::RunInfo();
    
    // set configuration parameters
    //configuration.input_file_name = "../real_nets/drosophila.txt";
    configuration.input_file_name = "test_files/sistema_0_500.txt";
    //configuration.rand_seed = 123456;
    //configuration.hs_count=1000;

    //configuration.hs_input_file_name = "tesths.txt";

    // create application object
    dci::Application* app = new dci::Application(configuration);

    // initialize application
    app->Init();
    
    
    // ********************************************************************* //
    // COMPUTATION SECTION - repeat as needed                                //
    // ********************************************************************* //
    
    // allocate memory for clusters
    vector<register_t*> clusters(88);
    
    // allocate memory for cluster indexes
    vector<float> output(88);


vector<unsigned int> cluster0 = {455, 478, 454, 456, 485, 476, 477, 471, 472};
vector<unsigned int> cluster1 = {277, 299, 279, 293, 268, 269, 261, 292};
vector<unsigned int> cluster2 = {69, 96, 53, 71, 60, 68, 74};
vector<unsigned int> cluster3 = {150, 165, 198, 461};
vector<unsigned int> cluster4 = {160, 199, 183, 171, 182, 195, 167, 196, 172, 189};
vector<unsigned int> cluster5 = {270, 271, 280, 281, 255, 287, 291, 344};
vector<unsigned int> cluster6 = {245, 246, 218, 231, 220, 224, 236};
vector<unsigned int> cluster7 = {406, 408, 425, 426, 412, 413};
vector<unsigned int> cluster8 = {108, 110, 129, 130, 100, 149, 102};
vector<unsigned int> cluster9 = {341, 428, 310, 317, 91};
vector<unsigned int> cluster10 = {210, 217, 225, 223, 232, 235, 201, 212, 209, 229};
vector<unsigned int> cluster11 = {162, 187, 191, 192, 155, 186, 193};
vector<unsigned int> cluster12 = {451, 491, 496, 483, 489, 487, 499};
vector<unsigned int> cluster13 = {156, 177, 190, 180, 152, 169, 174};
vector<unsigned int> cluster14 = {338, 349, 321, 322, 323, 324, 347};
vector<unsigned int> cluster15 = {384, 393, 372, 373, 374, 399, 368, 385};
vector<unsigned int> cluster16 = {354, 355, 367, 383, 498};
vector<unsigned int> cluster17 = {222, 242, 200, 248, 247};
vector<unsigned int> cluster18 = {407, 418, 444, 435, 439, 440};
vector<unsigned int> cluster19 = {55, 65, 58, 66, 92, 86};
vector<unsigned int> cluster20 = {51, 89, 61, 90, 50, 82, 85};
vector<unsigned int> cluster21 = {252, 256, 263, 346};
vector<unsigned int> cluster22 = {254, 257, 298, 286};
vector<unsigned int> cluster23 = {151, 159, 158, 168, 164, 185};
vector<unsigned int> cluster24 = {363, 375, 362, 387, 360, 386, 352, 364};
vector<unsigned int> cluster25 = {475, 490, 495, 452, 459, 484};
vector<unsigned int> cluster26 = {262, 266, 264, 259, 267, 273};
vector<unsigned int> cluster27 = {20, 34, 22, 24, 15, 38};
vector<unsigned int> cluster28 = {303, 308, 301, 333, 305, 331, 309, 332};
vector<unsigned int> cluster29 = {64, 81, 72, 73, 80, 83, 87};
vector<unsigned int> cluster30 = {2, 37, 19, 39, 40, 9, 16, 36};
vector<unsigned int> cluster31 = {115, 136, 126, 131, 415};
vector<unsigned int> cluster32 = {304, 320, 307, 312, 430, 438};
vector<unsigned int> cluster33 = {59, 62, 78, 63, 54, 67, 70};
vector<unsigned int> cluster34 = {405, 421, 334, 335, 424, 427, 448};
vector<unsigned int> cluster35 = {103, 105, 147, 114, 141, 148};
vector<unsigned int> cluster36 = {206, 249, 205, 234, 238, 241};
vector<unsigned int> cluster37 = {416, 417, 446, 449, 400, 404, 445};
vector<unsigned int> cluster38 = {376, 389, 366, 381, 378, 388, 394};
vector<unsigned int> cluster39 = {139, 431, 127, 457, 480};
vector<unsigned int> cluster40 = {482, 497, 458, 453, 473};
vector<unsigned int> cluster41 = {93, 99, 94, 52, 76, 77};
vector<unsigned int> cluster42 = {124, 132, 135, 107, 137, 138};
vector<unsigned int> cluster43 = {175, 194, 166, 170, 173};
vector<unsigned int> cluster44 = {5, 10, 25, 0, 33, 42};
vector<unsigned int> cluster45 = {465, 468, 493, 460, 486, 494};
vector<unsigned int> cluster46 = {14, 30, 27, 13, 26, 29};
vector<unsigned int> cluster47 = {353, 391, 351, 392, 350, 380, 390};
vector<unsigned int> cluster48 = {302, 313, 337, 342, 345, 348};
vector<unsigned int> cluster49 = {276, 288, 296, 260, 278};
vector<unsigned int> cluster50 = {113, 140, 123, 116, 118, 144};
vector<unsigned int> cluster51 = {358, 370, 359, 357, 365, 356, 377};
vector<unsigned int> cluster52 = {203, 240, 207, 202, 228, 239};
vector<unsigned int> cluster53 = {403, 411, 423, 419, 420, 422};
vector<unsigned int> cluster54 = {434, 441, 443, 433, 436, 442};
vector<unsigned int> cluster55 = {122, 142, 112, 101, 109, 117};
vector<unsigned int> cluster56 = {214, 233, 219, 221, 227};
vector<unsigned int> cluster57 = {325, 343, 44, 306, 437};
vector<unsigned int> cluster58 = {272, 294, 295, 251};
vector<unsigned int> cluster59 = {361, 369, 371, 379, 398};
vector<unsigned int> cluster60 = {35, 45, 12, 43, 11, 46, 49};
vector<unsigned int> cluster61 = {125, 128, 133, 111, 143};
vector<unsigned int> cluster62 = {450, 488, 492, 474, 479};
vector<unsigned int> cluster63 = {157, 181, 179, 184, 188, 197};
vector<unsigned int> cluster64 = {410, 447, 401, 409};
vector<unsigned int> cluster65 = {402, 414, 429};
vector<unsigned int> cluster66 = {275, 297, 285, 250, 290};
vector<unsigned int> cluster67 = {237, 244, 211, 230, 243};
vector<unsigned int> cluster68 = {17, 31, 18, 21, 23};
vector<unsigned int> cluster69 = {75, 84, 56, 57, 79};
vector<unsigned int> cluster70 = {463, 481, 462, 464};
vector<unsigned int> cluster71 = {282, 283, 253, 258};
vector<unsigned int> cluster72 = {1, 47, 6, 48, 3, 41};
vector<unsigned int> cluster73 = {311, 432, 315, 326};
vector<unsigned int> cluster74 = {104, 146, 106, 120, 134};
vector<unsigned int> cluster75 = {284, 289, 265, 274};
vector<unsigned int> cluster76 = {329, 330, 314, 316};
vector<unsigned int> cluster77 = {154, 178, 176};
vector<unsigned int> cluster78 = {469, 470, 466, 467};
vector<unsigned int> cluster79 = {7, 8, 4, 28, 32};
vector<unsigned int> cluster80 = {318, 339, 300, 327};
vector<unsigned int> cluster81 = {319, 336, 328, 340};
vector<unsigned int> cluster82 = {382, 395, 396, 397};
vector<unsigned int> cluster83 = {95, 97, 88, 98};
vector<unsigned int> cluster84 = {153, 161, 163};
vector<unsigned int> cluster85 = {215, 216, 208};
vector<unsigned int> cluster86 = {119, 121, 145};
vector<unsigned int> cluster87 = {204, 213, 226};


clusters[0] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[1] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[2] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[3] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[4] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[5] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[6] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[7] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[8] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[9] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[10] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[11] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[12] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[13] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[14] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[15] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[16] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[17] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[18] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[19] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[20] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[21] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[22] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[23] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[24] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[25] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[26] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[27] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[28] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[29] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[30] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[31] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[32] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[33] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[34] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[35] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[36] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[37] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[38] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[39] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[40] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[41] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[42] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[43] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[44] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[45] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[46] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[47] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[48] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[49] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[50] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[51] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[52] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[53] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[54] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[55] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[56] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[57] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[58] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[59] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[60] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[61] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[62] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[63] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[64] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[65] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[66] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[67] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[68] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[69] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[70] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[71] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[72] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[73] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[74] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[75] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[76] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[77] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[78] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[79] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[80] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[81] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[82] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[83] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[84] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[85] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[86] = (register_t*)malloc(app->getAgentSizeInBytes());
clusters[87] = (register_t*)malloc(app->getAgentSizeInBytes());


dci::ClusterUtils::setClusterFromPosArray(clusters[0], cluster0, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[1], cluster1, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[2], cluster2, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[3], cluster3, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[4], cluster4, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[5], cluster5, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[6], cluster6, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[7], cluster7, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[8], cluster8, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[9], cluster9, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[10], cluster10, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[11], cluster11, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[12], cluster12, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[13], cluster13, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[14], cluster14, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[15], cluster15, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[16], cluster16, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[17], cluster17, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[18], cluster18, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[19], cluster19, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[20], cluster20, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[21], cluster21, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[22], cluster22, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[23], cluster23, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[24], cluster24, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[25], cluster25, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[26], cluster26, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[27], cluster27, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[28], cluster28, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[29], cluster29, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[30], cluster30, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[31], cluster31, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[32], cluster32, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[33], cluster33, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[34], cluster34, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[35], cluster35, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[36], cluster36, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[37], cluster37, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[38], cluster38, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[39], cluster39, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[40], cluster40, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[41], cluster41, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[42], cluster42, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[43], cluster43, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[44], cluster44, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[45], cluster45, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[46], cluster46, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[47], cluster47, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[48], cluster48, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[49], cluster49, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[50], cluster50, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[51], cluster51, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[52], cluster52, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[53], cluster53, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[54], cluster54, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[55], cluster55, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[56], cluster56, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[57], cluster57, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[58], cluster58, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[59], cluster59, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[60], cluster60, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[61], cluster61, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[62], cluster62, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[63], cluster63, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[64], cluster64, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[65], cluster65, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[66], cluster66, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[67], cluster67, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[68], cluster68, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[69], cluster69, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[70], cluster70, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[71], cluster71, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[72], cluster72, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[73], cluster73, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[74], cluster74, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[75], cluster75, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[76], cluster76, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[77], cluster77, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[78], cluster78, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[79], cluster79, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[80], cluster80, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[81], cluster81, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[82], cluster82, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[83], cluster83, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[84], cluster84, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[85], cluster85, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[86], cluster86, app->getNumberOfAgents());
dci::ClusterUtils::setClusterFromPosArray(clusters[87], cluster87, app->getNumberOfAgents());



    // perform computation
    app->ComputeIndex(clusters, output);
    
    // print clusters and results
for(int i=0; i<clusters.size(); i++)
{
    //dci::ClusterUtils::print(cout, clusters[0], app->getNumberOfAgents());
    cout << output[i] << endl;
}
    
    
    // free memory
for(int i=0; i<clusters.size(); i++)
{

    free(clusters[i]);
}
    
    
    // ********************************************************************* //
    // SHUTDOWN SECTION - call once before exit                              //
    // ********************************************************************* //
    
    // delete app object
    delete app;
    
    return 0;
    
}

