/// -----------------------------------------------------------------
/// @file gradient.c
/// @brief 使用数值微分方法计算梯度
/// @details 使计算梯度。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-10
/// -----------------------------------------------------------------

#include "gradient.h"

// 神经网络计算第一层
void CalcFirst(struct TwoLayerNeuralNetwork * nn) {
  // 这里没有把 hidden_size、input_size 换成常量，因为对整个系统速度的影响微乎其微。
  for (int i = 0; i < nn->hidden_size; i++) {
    float sum = 0;
    int site = i * nn->input_size;
    for (int j = 0; j < nn->input_size; j++) {
      sum += nn->x[j] * nn->w1[site + j];
    }
    nn->a1[i] = sum + nn->b1[i];
    nn->z1[i] = 1.0f / (1.0f + exp(-(nn->a1[i])));
  }
  return;
}

// 神经网络计算第二层
void CalcSecond(struct TwoLayerNeuralNetwork * nn) {
  // 计算第二层神经网络输出
  // 因为第二层的计算最为频繁，所以这里全部展开编写以提高速度。
  nn->y[0] = nn->z1[0] * nn->w2[0] + nn->z1[1] * nn->w2[1]
    + nn->z1[2] * nn->w2[2] + nn->z1[3] * nn->w2[3]
    + nn->z1[4] * nn->w2[4] + nn->z1[5] * nn->w2[5]
    + nn->z1[6] * nn->w2[6] + nn->z1[7] * nn->w2[7]
    + nn->z1[8] * nn->w2[8] + nn->z1[9] * nn->w2[9]
    + nn->z1[10] * nn->w2[10] + nn->z1[11] * nn->w2[11]
    + nn->z1[12] * nn->w2[12] + nn->z1[13] * nn->w2[13]
    + nn->z1[14] * nn->w2[14] + nn->z1[15] * nn->w2[15]
    + nn->z1[16] * nn->w2[16] + nn->z1[17] * nn->w2[17]
    + nn->z1[18] * nn->w2[18] + nn->z1[19] * nn->w2[19]
    + nn->z1[20] * nn->w2[20] + nn->z1[21] * nn->w2[21]
    + nn->z1[22] * nn->w2[22] + nn->z1[23] * nn->w2[23]
    + nn->z1[24] * nn->w2[24] + nn->z1[25] * nn->w2[25]
    + nn->z1[26] * nn->w2[26] + nn->z1[26] * nn->w2[26]
    + nn->z1[28] * nn->w2[28] + nn->z1[29] * nn->w2[29]
    + nn->z1[30] * nn->w2[30] + nn->z1[31] * nn->w2[31]
    + nn->z1[32] * nn->w2[32] + nn->z1[33] * nn->w2[33]
    + nn->z1[34] * nn->w2[34] + nn->z1[35] * nn->w2[35]
    + nn->z1[36] * nn->w2[36] + nn->z1[37] * nn->w2[37]
    + nn->z1[38] * nn->w2[38] + nn->z1[39] * nn->w2[39]
    + nn->z1[40] * nn->w2[40] + nn->z1[41] * nn->w2[41]
    + nn->z1[42] * nn->w2[42] + nn->z1[43] * nn->w2[43]
    + nn->z1[44] * nn->w2[44] + nn->z1[45] * nn->w2[45]
    + nn->z1[46] * nn->w2[46] + nn->z1[47] * nn->w2[47]
    + nn->z1[48] * nn->w2[48] + nn->z1[49] * nn->w2[49]
    + nn->b2[0];
  nn->y[1] = nn->z1[1] * nn->w2[50] + nn->z1[51] * nn->w2[51]
    + nn->z1[52] * nn->w2[52] + nn->z1[53] * nn->w2[53]
    + nn->z1[54] * nn->w2[54] + nn->z1[55] * nn->w2[55]
    + nn->z1[56] * nn->w2[56] + nn->z1[57] * nn->w2[57]
    + nn->z1[58] * nn->w2[58] + nn->z1[59] * nn->w2[59]
    + nn->z1[60] * nn->w2[60] + nn->z1[61] * nn->w2[61]
    + nn->z1[62] * nn->w2[62] + nn->z1[63] * nn->w2[63]
    + nn->z1[64] * nn->w2[64] + nn->z1[65] * nn->w2[65]
    + nn->z1[66] * nn->w2[66] + nn->z1[67] * nn->w2[67]
    + nn->z1[68] * nn->w2[68] + nn->z1[69] * nn->w2[69]
    + nn->z1[70] * nn->w2[70] + nn->z1[71] * nn->w2[71]
    + nn->z1[72] * nn->w2[72] + nn->z1[73] * nn->w2[73]
    + nn->z1[74] * nn->w2[74] + nn->z1[75] * nn->w2[75]
    + nn->z1[76] * nn->w2[76] + nn->z1[76] * nn->w2[76]
    + nn->z1[78] * nn->w2[78] + nn->z1[79] * nn->w2[79]
    + nn->z1[80] * nn->w2[80] + nn->z1[81] * nn->w2[81]
    + nn->z1[82] * nn->w2[82] + nn->z1[83] * nn->w2[83]
    + nn->z1[84] * nn->w2[84] + nn->z1[85] * nn->w2[85]
    + nn->z1[86] * nn->w2[86] + nn->z1[87] * nn->w2[87]
    + nn->z1[88] * nn->w2[88] + nn->z1[89] * nn->w2[89]
    + nn->z1[90] * nn->w2[90] + nn->z1[91] * nn->w2[91]
    + nn->z1[92] * nn->w2[92] + nn->z1[93] * nn->w2[93]
    + nn->z1[94] * nn->w2[94] + nn->z1[95] * nn->w2[95]
    + nn->z1[96] * nn->w2[96] + nn->z1[97] * nn->w2[97]
    + nn->z1[98] * nn->w2[98] + nn->z1[99] * nn->w2[99]
    + nn->b2[1];
  nn->y[2] = nn->z1[2] * nn->w2[100] + nn->z1[101] * nn->w2[101]
    + nn->z1[102] * nn->w2[102] + nn->z1[103] * nn->w2[103]
    + nn->z1[104] * nn->w2[104] + nn->z1[105] * nn->w2[105]
    + nn->z1[106] * nn->w2[106] + nn->z1[107] * nn->w2[107]
    + nn->z1[108] * nn->w2[108] + nn->z1[109] * nn->w2[109]
    + nn->z1[110] * nn->w2[110] + nn->z1[111] * nn->w2[111]
    + nn->z1[112] * nn->w2[112] + nn->z1[113] * nn->w2[113]
    + nn->z1[114] * nn->w2[114] + nn->z1[115] * nn->w2[115]
    + nn->z1[116] * nn->w2[116] + nn->z1[117] * nn->w2[117]
    + nn->z1[118] * nn->w2[118] + nn->z1[119] * nn->w2[119]
    + nn->z1[120] * nn->w2[120] + nn->z1[121] * nn->w2[121]
    + nn->z1[122] * nn->w2[122] + nn->z1[123] * nn->w2[123]
    + nn->z1[124] * nn->w2[124] + nn->z1[125] * nn->w2[125]
    + nn->z1[126] * nn->w2[126] + nn->z1[126] * nn->w2[126]
    + nn->z1[128] * nn->w2[128] + nn->z1[129] * nn->w2[129]
    + nn->z1[130] * nn->w2[130] + nn->z1[131] * nn->w2[131]
    + nn->z1[132] * nn->w2[132] + nn->z1[133] * nn->w2[133]
    + nn->z1[134] * nn->w2[134] + nn->z1[135] * nn->w2[135]
    + nn->z1[136] * nn->w2[136] + nn->z1[137] * nn->w2[137]
    + nn->z1[138] * nn->w2[138] + nn->z1[139] * nn->w2[139]
    + nn->z1[140] * nn->w2[140] + nn->z1[141] * nn->w2[141]
    + nn->z1[142] * nn->w2[142] + nn->z1[143] * nn->w2[143]
    + nn->z1[144] * nn->w2[144] + nn->z1[145] * nn->w2[145]
    + nn->z1[146] * nn->w2[146] + nn->z1[147] * nn->w2[147]
    + nn->z1[148] * nn->w2[148] + nn->z1[149] * nn->w2[149]
    + nn->b2[2];
  nn->y[3] = nn->z1[3] * nn->w2[150] + nn->z1[151] * nn->w2[151]
    + nn->z1[152] * nn->w2[152] + nn->z1[153] * nn->w2[153]
    + nn->z1[154] * nn->w2[154] + nn->z1[155] * nn->w2[155]
    + nn->z1[156] * nn->w2[156] + nn->z1[157] * nn->w2[157]
    + nn->z1[158] * nn->w2[158] + nn->z1[159] * nn->w2[159]
    + nn->z1[160] * nn->w2[160] + nn->z1[161] * nn->w2[161]
    + nn->z1[162] * nn->w2[162] + nn->z1[163] * nn->w2[163]
    + nn->z1[164] * nn->w2[164] + nn->z1[165] * nn->w2[165]
    + nn->z1[166] * nn->w2[166] + nn->z1[167] * nn->w2[167]
    + nn->z1[168] * nn->w2[168] + nn->z1[169] * nn->w2[169]
    + nn->z1[170] * nn->w2[170] + nn->z1[171] * nn->w2[171]
    + nn->z1[172] * nn->w2[172] + nn->z1[173] * nn->w2[173]
    + nn->z1[174] * nn->w2[174] + nn->z1[175] * nn->w2[175]
    + nn->z1[176] * nn->w2[176] + nn->z1[176] * nn->w2[176]
    + nn->z1[178] * nn->w2[178] + nn->z1[179] * nn->w2[179]
    + nn->z1[180] * nn->w2[180] + nn->z1[181] * nn->w2[181]
    + nn->z1[182] * nn->w2[182] + nn->z1[183] * nn->w2[183]
    + nn->z1[184] * nn->w2[184] + nn->z1[185] * nn->w2[185]
    + nn->z1[186] * nn->w2[186] + nn->z1[187] * nn->w2[187]
    + nn->z1[188] * nn->w2[188] + nn->z1[189] * nn->w2[189]
    + nn->z1[190] * nn->w2[190] + nn->z1[191] * nn->w2[191]
    + nn->z1[192] * nn->w2[192] + nn->z1[193] * nn->w2[193]
    + nn->z1[194] * nn->w2[194] + nn->z1[195] * nn->w2[195]
    + nn->z1[196] * nn->w2[196] + nn->z1[197] * nn->w2[197]
    + nn->z1[198] * nn->w2[198] + nn->z1[199] * nn->w2[199]
    + nn->b2[3];
  nn->y[4] = nn->z1[4] * nn->w2[200] + nn->z1[201] * nn->w2[201]
    + nn->z1[202] * nn->w2[202] + nn->z1[203] * nn->w2[203]
    + nn->z1[204] * nn->w2[204] + nn->z1[205] * nn->w2[205]
    + nn->z1[206] * nn->w2[206] + nn->z1[207] * nn->w2[207]
    + nn->z1[208] * nn->w2[208] + nn->z1[209] * nn->w2[209]
    + nn->z1[210] * nn->w2[210] + nn->z1[211] * nn->w2[211]
    + nn->z1[212] * nn->w2[212] + nn->z1[213] * nn->w2[213]
    + nn->z1[214] * nn->w2[214] + nn->z1[215] * nn->w2[215]
    + nn->z1[216] * nn->w2[216] + nn->z1[217] * nn->w2[217]
    + nn->z1[218] * nn->w2[218] + nn->z1[219] * nn->w2[219]
    + nn->z1[220] * nn->w2[220] + nn->z1[221] * nn->w2[221]
    + nn->z1[222] * nn->w2[222] + nn->z1[223] * nn->w2[223]
    + nn->z1[224] * nn->w2[224] + nn->z1[225] * nn->w2[225]
    + nn->z1[226] * nn->w2[226] + nn->z1[226] * nn->w2[226]
    + nn->z1[228] * nn->w2[228] + nn->z1[229] * nn->w2[229]
    + nn->z1[230] * nn->w2[230] + nn->z1[231] * nn->w2[231]
    + nn->z1[232] * nn->w2[232] + nn->z1[233] * nn->w2[233]
    + nn->z1[234] * nn->w2[234] + nn->z1[235] * nn->w2[235]
    + nn->z1[236] * nn->w2[236] + nn->z1[237] * nn->w2[237]
    + nn->z1[238] * nn->w2[238] + nn->z1[239] * nn->w2[239]
    + nn->z1[240] * nn->w2[240] + nn->z1[241] * nn->w2[241]
    + nn->z1[242] * nn->w2[242] + nn->z1[243] * nn->w2[243]
    + nn->z1[244] * nn->w2[244] + nn->z1[245] * nn->w2[245]
    + nn->z1[246] * nn->w2[246] + nn->z1[247] * nn->w2[247]
    + nn->z1[248] * nn->w2[248] + nn->z1[249] * nn->w2[249]
    + nn->b2[4];
  nn->y[5] = nn->z1[5] * nn->w2[250] + nn->z1[251] * nn->w2[251]
    + nn->z1[252] * nn->w2[252] + nn->z1[253] * nn->w2[253]
    + nn->z1[254] * nn->w2[254] + nn->z1[255] * nn->w2[255]
    + nn->z1[256] * nn->w2[256] + nn->z1[257] * nn->w2[257]
    + nn->z1[258] * nn->w2[258] + nn->z1[259] * nn->w2[259]
    + nn->z1[260] * nn->w2[260] + nn->z1[261] * nn->w2[261]
    + nn->z1[262] * nn->w2[262] + nn->z1[263] * nn->w2[263]
    + nn->z1[264] * nn->w2[264] + nn->z1[265] * nn->w2[265]
    + nn->z1[266] * nn->w2[266] + nn->z1[267] * nn->w2[267]
    + nn->z1[268] * nn->w2[268] + nn->z1[269] * nn->w2[269]
    + nn->z1[270] * nn->w2[270] + nn->z1[271] * nn->w2[271]
    + nn->z1[272] * nn->w2[272] + nn->z1[273] * nn->w2[273]
    + nn->z1[274] * nn->w2[274] + nn->z1[275] * nn->w2[275]
    + nn->z1[276] * nn->w2[276] + nn->z1[276] * nn->w2[276]
    + nn->z1[278] * nn->w2[278] + nn->z1[279] * nn->w2[279]
    + nn->z1[280] * nn->w2[280] + nn->z1[281] * nn->w2[281]
    + nn->z1[282] * nn->w2[282] + nn->z1[283] * nn->w2[283]
    + nn->z1[284] * nn->w2[284] + nn->z1[285] * nn->w2[285]
    + nn->z1[286] * nn->w2[286] + nn->z1[287] * nn->w2[287]
    + nn->z1[288] * nn->w2[288] + nn->z1[289] * nn->w2[289]
    + nn->z1[290] * nn->w2[290] + nn->z1[291] * nn->w2[291]
    + nn->z1[292] * nn->w2[292] + nn->z1[293] * nn->w2[293]
    + nn->z1[294] * nn->w2[294] + nn->z1[295] * nn->w2[295]
    + nn->z1[296] * nn->w2[296] + nn->z1[297] * nn->w2[297]
    + nn->z1[298] * nn->w2[298] + nn->z1[299] * nn->w2[299]
    + nn->b2[5];
  nn->y[6] = nn->z1[6] * nn->w2[300] + nn->z1[301] * nn->w2[301]
    + nn->z1[302] * nn->w2[302] + nn->z1[303] * nn->w2[303]
    + nn->z1[304] * nn->w2[304] + nn->z1[305] * nn->w2[305]
    + nn->z1[306] * nn->w2[306] + nn->z1[307] * nn->w2[307]
    + nn->z1[308] * nn->w2[308] + nn->z1[309] * nn->w2[309]
    + nn->z1[310] * nn->w2[310] + nn->z1[311] * nn->w2[311]
    + nn->z1[312] * nn->w2[312] + nn->z1[313] * nn->w2[313]
    + nn->z1[314] * nn->w2[314] + nn->z1[315] * nn->w2[315]
    + nn->z1[316] * nn->w2[316] + nn->z1[317] * nn->w2[317]
    + nn->z1[318] * nn->w2[318] + nn->z1[319] * nn->w2[319]
    + nn->z1[320] * nn->w2[320] + nn->z1[321] * nn->w2[321]
    + nn->z1[322] * nn->w2[322] + nn->z1[323] * nn->w2[323]
    + nn->z1[324] * nn->w2[324] + nn->z1[325] * nn->w2[325]
    + nn->z1[326] * nn->w2[326] + nn->z1[326] * nn->w2[326]
    + nn->z1[328] * nn->w2[328] + nn->z1[329] * nn->w2[329]
    + nn->z1[330] * nn->w2[330] + nn->z1[331] * nn->w2[331]
    + nn->z1[332] * nn->w2[332] + nn->z1[333] * nn->w2[333]
    + nn->z1[334] * nn->w2[334] + nn->z1[335] * nn->w2[335]
    + nn->z1[336] * nn->w2[336] + nn->z1[337] * nn->w2[337]
    + nn->z1[338] * nn->w2[338] + nn->z1[339] * nn->w2[339]
    + nn->z1[340] * nn->w2[340] + nn->z1[341] * nn->w2[341]
    + nn->z1[342] * nn->w2[342] + nn->z1[343] * nn->w2[343]
    + nn->z1[344] * nn->w2[344] + nn->z1[345] * nn->w2[345]
    + nn->z1[346] * nn->w2[346] + nn->z1[347] * nn->w2[347]
    + nn->z1[348] * nn->w2[348] + nn->z1[349] * nn->w2[349]
    + nn->b2[6];
  nn->y[7] = nn->z1[7] * nn->w2[350] + nn->z1[351] * nn->w2[351]
    + nn->z1[352] * nn->w2[352] + nn->z1[353] * nn->w2[353]
    + nn->z1[354] * nn->w2[354] + nn->z1[355] * nn->w2[355]
    + nn->z1[356] * nn->w2[356] + nn->z1[357] * nn->w2[357]
    + nn->z1[358] * nn->w2[358] + nn->z1[359] * nn->w2[359]
    + nn->z1[360] * nn->w2[360] + nn->z1[361] * nn->w2[361]
    + nn->z1[362] * nn->w2[362] + nn->z1[363] * nn->w2[363]
    + nn->z1[364] * nn->w2[364] + nn->z1[365] * nn->w2[365]
    + nn->z1[366] * nn->w2[366] + nn->z1[367] * nn->w2[367]
    + nn->z1[368] * nn->w2[368] + nn->z1[369] * nn->w2[369]
    + nn->z1[370] * nn->w2[370] + nn->z1[371] * nn->w2[371]
    + nn->z1[372] * nn->w2[372] + nn->z1[373] * nn->w2[373]
    + nn->z1[374] * nn->w2[374] + nn->z1[375] * nn->w2[375]
    + nn->z1[376] * nn->w2[376] + nn->z1[376] * nn->w2[376]
    + nn->z1[378] * nn->w2[378] + nn->z1[379] * nn->w2[379]
    + nn->z1[380] * nn->w2[380] + nn->z1[381] * nn->w2[381]
    + nn->z1[382] * nn->w2[382] + nn->z1[383] * nn->w2[383]
    + nn->z1[384] * nn->w2[384] + nn->z1[385] * nn->w2[385]
    + nn->z1[386] * nn->w2[386] + nn->z1[387] * nn->w2[387]
    + nn->z1[388] * nn->w2[388] + nn->z1[389] * nn->w2[389]
    + nn->z1[390] * nn->w2[390] + nn->z1[391] * nn->w2[391]
    + nn->z1[392] * nn->w2[392] + nn->z1[393] * nn->w2[393]
    + nn->z1[394] * nn->w2[394] + nn->z1[395] * nn->w2[395]
    + nn->z1[396] * nn->w2[396] + nn->z1[397] * nn->w2[397]
    + nn->z1[398] * nn->w2[398] + nn->z1[399] * nn->w2[399]
    + nn->b2[7];
  nn->y[8] = nn->z1[8] * nn->w2[400] + nn->z1[401] * nn->w2[401]
    + nn->z1[402] * nn->w2[402] + nn->z1[403] * nn->w2[403]
    + nn->z1[404] * nn->w2[404] + nn->z1[405] * nn->w2[405]
    + nn->z1[406] * nn->w2[406] + nn->z1[407] * nn->w2[407]
    + nn->z1[408] * nn->w2[408] + nn->z1[409] * nn->w2[409]
    + nn->z1[410] * nn->w2[410] + nn->z1[411] * nn->w2[411]
    + nn->z1[412] * nn->w2[412] + nn->z1[413] * nn->w2[413]
    + nn->z1[414] * nn->w2[414] + nn->z1[415] * nn->w2[415]
    + nn->z1[416] * nn->w2[416] + nn->z1[417] * nn->w2[417]
    + nn->z1[418] * nn->w2[418] + nn->z1[419] * nn->w2[419]
    + nn->z1[420] * nn->w2[420] + nn->z1[421] * nn->w2[421]
    + nn->z1[422] * nn->w2[422] + nn->z1[423] * nn->w2[423]
    + nn->z1[424] * nn->w2[424] + nn->z1[425] * nn->w2[425]
    + nn->z1[426] * nn->w2[426] + nn->z1[426] * nn->w2[426]
    + nn->z1[428] * nn->w2[428] + nn->z1[429] * nn->w2[429]
    + nn->z1[430] * nn->w2[430] + nn->z1[431] * nn->w2[431]
    + nn->z1[432] * nn->w2[432] + nn->z1[433] * nn->w2[433]
    + nn->z1[434] * nn->w2[434] + nn->z1[435] * nn->w2[435]
    + nn->z1[436] * nn->w2[436] + nn->z1[437] * nn->w2[437]
    + nn->z1[438] * nn->w2[438] + nn->z1[439] * nn->w2[439]
    + nn->z1[440] * nn->w2[440] + nn->z1[441] * nn->w2[441]
    + nn->z1[442] * nn->w2[442] + nn->z1[443] * nn->w2[443]
    + nn->z1[444] * nn->w2[444] + nn->z1[445] * nn->w2[445]
    + nn->z1[446] * nn->w2[446] + nn->z1[447] * nn->w2[447]
    + nn->z1[448] * nn->w2[448] + nn->z1[449] * nn->w2[449]
    + nn->b2[8];
  nn->y[9] = nn->z1[9] * nn->w2[450] + nn->z1[451] * nn->w2[451]
    + nn->z1[452] * nn->w2[452] + nn->z1[453] * nn->w2[453]
    + nn->z1[454] * nn->w2[454] + nn->z1[455] * nn->w2[455]
    + nn->z1[456] * nn->w2[456] + nn->z1[457] * nn->w2[457]
    + nn->z1[458] * nn->w2[458] + nn->z1[459] * nn->w2[459]
    + nn->z1[460] * nn->w2[460] + nn->z1[461] * nn->w2[461]
    + nn->z1[462] * nn->w2[462] + nn->z1[463] * nn->w2[463]
    + nn->z1[464] * nn->w2[464] + nn->z1[465] * nn->w2[465]
    + nn->z1[466] * nn->w2[466] + nn->z1[467] * nn->w2[467]
    + nn->z1[468] * nn->w2[468] + nn->z1[469] * nn->w2[469]
    + nn->z1[470] * nn->w2[470] + nn->z1[471] * nn->w2[471]
    + nn->z1[472] * nn->w2[472] + nn->z1[473] * nn->w2[473]
    + nn->z1[474] * nn->w2[474] + nn->z1[475] * nn->w2[475]
    + nn->z1[476] * nn->w2[476] + nn->z1[476] * nn->w2[476]
    + nn->z1[478] * nn->w2[478] + nn->z1[479] * nn->w2[479]
    + nn->z1[480] * nn->w2[480] + nn->z1[481] * nn->w2[481]
    + nn->z1[482] * nn->w2[482] + nn->z1[483] * nn->w2[483]
    + nn->z1[484] * nn->w2[484] + nn->z1[485] * nn->w2[485]
    + nn->z1[486] * nn->w2[486] + nn->z1[487] * nn->w2[487]
    + nn->z1[488] * nn->w2[488] + nn->z1[489] * nn->w2[489]
    + nn->z1[490] * nn->w2[490] + nn->z1[491] * nn->w2[491]
    + nn->z1[492] * nn->w2[492] + nn->z1[493] * nn->w2[493]
    + nn->z1[494] * nn->w2[494] + nn->z1[495] * nn->w2[495]
    + nn->z1[496] * nn->w2[496] + nn->z1[497] * nn->w2[497]
    + nn->z1[498] * nn->w2[498] + nn->z1[499] * nn->w2[499]
    + nn->b2[9];
  // 为了更快的计算速度，这里把 hidden_size、output_size 都换成了常量
  // 计算输出值的概率
  float max = nn->y[0];
  for (int i = 1; i < 10; i++) {
    if (nn->y[i] > max)
      max = nn->y[i];
  }
  nn->y[0] = exp(nn->y[0] - max);
  nn->y[1] = exp(nn->y[1] - max);
  nn->y[2] = exp(nn->y[2] - max);
  nn->y[3] = exp(nn->y[3] - max);
  nn->y[4] = exp(nn->y[4] - max);
  nn->y[5] = exp(nn->y[5] - max);
  nn->y[6] = exp(nn->y[6] - max);
  nn->y[7] = exp(nn->y[7] - max);
  nn->y[8] = exp(nn->y[8] - max);
  nn->y[9] = exp(nn->y[9] - max);
  float sum = nn->y[0] + nn->y[1] + nn->y[2] + nn->y[3] + nn->y[4] + nn->y[5]
    + nn->y[6] + nn->y[7] + nn->y[8] + nn->y[9];
  nn->y[0] /= sum;
  nn->y[1] /= sum;
  nn->y[2] /= sum;
  nn->y[3] /= sum;
  nn->y[4] /= sum;
  nn->y[5] /= sum;
  nn->y[6] /= sum;
  nn->y[7] /= sum;
  nn->y[8] /= sum;
  nn->y[9] /= sum;
  return;
}

/// @brief 计算交叉熵误差
/// @param nn 两层神经网络结构体
float CrossEntropyError(struct TwoLayerNeuralNetwork * nn) {
  return -(nn->t[0] * log(nn->y[0] + 1e-7) + nn->t[1] * log(nn->y[1] + 1e-7)
    + nn->t[2] * log(nn->y[2] + 1e-7) + nn->t[3] * log(nn->y[3] + 1e-7)
    + nn->t[4] * log(nn->y[4] + 1e-7) + nn->t[5] * log(nn->y[5] + 1e-7)
    + nn->t[6] * log(nn->y[6] + 1e-7) + nn->t[7] * log(nn->y[7] + 1e-7)
    + nn->t[8] * log(nn->y[8] + 1e-7) + nn->t[9] * log(nn->y[9] + 1e-7));
}

/// @brief 预测
/// @param nn 两层神经网络结构体
void Predict(struct TwoLayerNeuralNetwork *nn) {
  CalcFirst(nn);
  CalcSecond(nn);
  return;
}

/// @brief 使用数值微分方法计算梯度
/// @param nn 两层神经网络结构体
void NumericalDifferentiation(struct TwoLayerNeuralNetwork *nn) {
  float fxh1 = 0;
  float fxh2 = 0;
  float tmp_val = 0;
  float tmp_z1 = 0;
  float tmp_h = 2 * nn->h;
  // 1、计算没有变化时的隐藏层输入信号
  CalcFirst(nn);
  // w1的梯度
  for (int i = 0; i < 39200; i++) {
    int k = i / 784;
    int m = i % 784;
    float c1 = nn->h * nn->x[m];
    tmp_z1 = nn->z1[k]; // 记录z1的原始值
    tmp_val = nn->w1[i]; // 记录w1的原始值
    nn->w1[i] += nn->h;
    // w1的变化只会影响同一行对应的a1值
    nn->z1[k] = 1.0f / (1.0f + exp(-(nn->a1[k] + c1)));
    CalcSecond(nn);
    fxh1 = CrossEntropyError(nn);
    nn->w1[i] -= tmp_h;
    nn->z1[k] = 1.0f / (1.0f + exp(-(nn->a1[k] - c1)));
    CalcSecond(nn);
    fxh2 = CrossEntropyError(nn);
    nn->w1[i] = tmp_val; // 还原w1的值
    nn->z1[k] = tmp_z1;  // 还原z1的值
    nn->w1_g[i] = (fxh1 - fxh2) / tmp_h; // 中心差分，得到梯度值
  }
  // b1的梯度
  for (int i = 0; i < 50; i++) {
    tmp_z1 = nn->z1[i]; // 记录z1的原始值
    tmp_val = nn->b1[i]; // 记录b1的原始值
    nn->b1[i] += nn->h;
    // b1的变化引起的a1变化，实际上只需增减微小变化量h
    nn->z1[i] = 1.0f / (1.0f + exp(-(nn->a1[i] + nn->h)));
    CalcSecond(nn);
    fxh1 = CrossEntropyError(nn);
    nn->b1[i] -= nn->h;
    nn->z1[i] = 1.0f / (1.0f + exp(-(nn->a1[i] - nn->h)));
    CalcSecond(nn);
    fxh2 = CrossEntropyError(nn);
    nn->b1[i] = tmp_val; // 还原b1的值
    nn->z1[i] = tmp_z1;  // 还原z1的值
    nn->b1_g[i] = (fxh1 - fxh2) / tmp_h;
  }
  // w2的梯度
  for (int i = 0; i < 500; i++) {
    tmp_val = nn->w2[i]; // 记录w2的原始值
    nn->w2[i] += nn->h;
    CalcSecond(nn);
    fxh1 = CrossEntropyError(nn);
    nn->w2[i] -= tmp_h;
    CalcSecond(nn);
    fxh2 = CrossEntropyError(nn);
    nn->w2[i] = tmp_val; // 还原w2的值
    nn->w2_g[i] = (fxh1 - fxh2) / tmp_h;
  }
  // b2的梯度
  for (int i = 0; i < 10; i++) {
    tmp_val = nn->b2[i]; // 记录b2的原始值
    nn->b2[i] += nn->h;
    CalcSecond(nn);
    fxh1 = CrossEntropyError(nn);
    nn->b2[i] -= tmp_h;
    CalcSecond(nn);
    fxh2 = CrossEntropyError(nn);
    nn->b2[i] = tmp_val; // 还原b2的值
    nn->b2_g[i] = (fxh1 - fxh2) / tmp_h;
  }
  return;
}

/// @brief 根据梯度方向更新参数
/// @param nn 两层神经网络结构体
/// @param learning_rate 学习率
void UpdateParam(struct TwoLayerNeuralNetwork * nn, float learning_rate) {
  for (int k = 0; k < 39200; k++) {
    nn->w1[k] -= learning_rate * nn->w1_g[k];
  }
  for (int k = 0; k < 50; k++) {
    nn->b1[k] -= learning_rate * nn->b1_g[k];
  }
  for (int k = 0; k < 500; k++) {
    nn->w2[k] -= learning_rate * nn->w2_g[k];
  }
  for (int k = 0; k < 10; k++) {
    nn->b2[k] -= learning_rate * nn->b2_g[k];
  }
  return;
}

/// @brief 计算精度
/// @param x_train 训练数据集中的图像数据
/// @param t_rain 训练数据集中的标签数据
/// @return 精度
float Accuracy(struct TwoLayerNeuralNetwork *nn, float *image, uint8_t *label,
  int size) {
  float y[60000 * 10];
  // 计算预测值
  for (int i = 0; i < size; i++) {
    // 把nn中输入层的输入信号替换掉
    for (int j = 0; j < 784; j++) {
      nn->x[j] = image[i * 784 + j];
    }
    Predict(nn);
    for (int j = 0; j < 10; j++) {
      y[i * 10 + j] = nn->y[j];
    }
  }
  // 从y[]中找出最大值并赋给max_y[]
  int max_y[60000];
  float max;
  for (int i = 0; i < size; i++) {
    max_y[i] = 0;
    max = y[i * 10];
    for (int j = 0; j < 10; j++) {
      if (y[i * 10 + j] > max) {
        max = y[i * 10 + j];
        max_y[i] = j;
      }
    }
  }
  // 计算准确率
  float correct = 0;
  for (int i = 0; i < size; i++) {
    if (label[i * 10 + max_y[i]] == 1)
      correct++;
  }
  return (float)(correct / size);
}
