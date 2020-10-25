#ifndef __DUMP_H__
#define __DUMP_H__

#include "../utils/parameters.h"
#include "../utils/commandline_flags.h"
#include "../utils/utils.h"
#include "../frequent_pattern_mining/frequent_pattern_mining.h"
#include "../data/documents.h"
#include "../classification/feature_extraction.h"
#include "../classification/label_generation.h"
#include "../classification/predict_quality.h"
#include "../model_training/segmentation.h"
#include <cstdio>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <string>
#include <ctime>

namespace Dump
{

using FrequentPatternMining::Pattern;
using FrequentPatternMining::patterns;
using FrequentPatternMining::pattern2id;
using FrequentPatternMining::truthPatterns;

void loadSegmentationModel(const string& filename)
{
    FILE* in = tryOpen(filename, "rb");
    bool flag;
    Binary::read(in, flag);
    myAssert(ENABLE_POS_TAGGING == flag, "Model and configuration mismatch! whether ENABLE_POS_TAGGING?");
    Binary::read(in, Segmentation::penalty);

    if (flag) {
        cerr << "POS guided model loaded." << endl;
    } else {
        cerr << "Length penalty model loaded." << endl;
        cerr << "\tpenalty = " << Segmentation::penalty << endl;
    }

    // quality phrases & unigrams
    size_t cnt = 0;
    Binary::read(in, cnt);
    patterns.resize(cnt);
    for (size_t i = 0; i < cnt; ++ i) {
        patterns[i].load(in);
    }
    cerr << "# of loaded patterns = " << cnt << endl;

    Binary::read(in, cnt);
    truthPatterns.resize(cnt);
    for (size_t i = 0; i < cnt; ++ i) {
        truthPatterns[i].load(in);
    }
    cerr << "# of loaded truth patterns = " << cnt << endl;

    if (flag) {
        // POS Tag mapping
        Binary::read(in, cnt);
        Documents::posTag.resize(cnt);
        for (int i = 0; i < Documents::posTag.size(); ++ i) {
            Binary::read(in, Documents::posTag[i]);
            Documents::posTag2id[Documents::posTag[i]] = i;
        }
        // cerr << "pos tags loaded" << endl;

        // POS Tag Transition
        Binary::read(in, cnt);
        Segmentation::connect.resize(cnt);
        for (int i = 0; i < Segmentation::connect.size(); ++ i) {
            Segmentation::connect[i].resize(cnt);
            for (int j = 0; j < Segmentation::connect[i].size(); ++ j) {
                Binary::read(in, Segmentation::connect[i][j]);
            }
        }
        cerr << "POS transition matrix loaded" << endl;
    }

    fclose(in);
}

void dumpSegmentationModel(const string& filename)
{
    FILE* out = tryOpen(filename, "wb");
    Binary::write(out, ENABLE_POS_TAGGING);
    Binary::write(out, Segmentation::penalty);

    // quality phrases & unigrams
    size_t cnt = 0;
    for (size_t i = 0; i < patterns.size(); ++ i) {
        if (patterns[i].size() > 1 && patterns[i].currentFreq > 0 || patterns[i].size() == 1 && patterns[i].currentFreq > 0 && unigrams[patterns[i].tokens[0]] >= MIN_SUP) {
            ++ cnt;
        }
    }
    Binary::write(out, cnt);
    if (INTERMEDIATE) {
        cerr << "# of phrases dumped = " << cnt << endl;
    }
    for (size_t i = 0; i < patterns.size(); ++ i) {
        if (patterns[i].size() > 1 && patterns[i].currentFreq > 0 || patterns[i].size() == 1 && patterns[i].currentFreq > 0 && unigrams[patterns[i].tokens[0]] >= MIN_SUP) {
            patterns[i].dump(out);
        }
    }

    // truth
    if (INTERMEDIATE) {
        cerr << "# of truth dumped = " << truthPatterns.size() << endl;
    }
    Binary::write(out, truthPatterns.size());
    for (size_t i = 0; i < truthPatterns.size(); ++ i) {
        truthPatterns[i].dump(out);
    }

    // POS Tag mapping
    Binary::write(out, Documents::posTag.size());
    for (int i = 0; i < Documents::posTag.size(); ++ i) {
        Binary::write(out, Documents::posTag[i]);
    }

    // POS Tag Transition
    Binary::write(out, Segmentation::connect.size());
    for (int i = 0; i < Segmentation::connect.size(); ++ i) {
        for (int j = 0; j < Segmentation::connect[i].size(); ++ j) {
            Binary::write(out, Segmentation::connect[i][j]);
        }
    }

    fclose(out);
}

void dumpPOSTransition(const string& filename)
{
    FILE* out = tryOpen(filename, "w");
    for (int i = 0; i < Documents::posTag.size(); ++ i) {
        fprintf(out, "\t%s", Documents::posTag[i].c_str());
    }
    fprintf(out, "\n");
    for (int i = 0; i < Documents::posTag.size(); ++ i) {
        fprintf(out, "%s", Documents::posTag[i].c_str());
        for (int j = 0; j < Documents::posTag.size(); ++ j) {
            fprintf(out, "\t%.10f", Segmentation::connect[i][j]);
        }
        fprintf(out, "\n");
    }
    fclose(out);
}

void dumpFeatures(const string& filename, const vector<vector<double>>& features, const vector<Pattern>& truth)
{
    FILE* out = tryOpen(filename, "w");
    for (Pattern pattern : truth) {
        PATTERN_ID_TYPE i = FrequentPatternMining::pattern2id[pattern.hashValue];
        if (features[i].size() > 0) {
            for (int j = 0; j < features[i].size(); ++ j) {
                fprintf(out, "%.10f%c", features[i][j], j + 1 == features[i].size() ? '\n' : '\t');
            }
        }
    }
    fclose(out);
}

void dumpLabels(const string& filename, const vector<Pattern>& truth)
{
    FILE* out = tryOpen(filename, "w");
    for (Pattern pattern : truth) {
        for (int j = 0; j < pattern.tokens.size(); ++ j) {
            fprintf(out, "%d%c", pattern.tokens[j], j + 1 == pattern.tokens.size() ? '\n' : ' ');
        }
    }
    fclose(out);
}

template<class T>
void dumpRankingList(const string& filename, vector<pair<T, PATTERN_ID_TYPE>> &order)
{
    FILE* out = tryOpen(filename, "w");
    sort(order.rbegin(), order.rend());
    for (size_t iter = 0; iter < order.size(); ++ iter) {
        PATTERN_ID_TYPE i = order[iter].second;

        /**
         * Beidan made threshold
         */
        if(patterns[i].quality > 0.6){

            fprintf(out, "%.10f\t", patterns[i].quality);
            for (int j = 0; j < patterns[i].tokens.size(); ++ j) {
                fprintf(out, "%d%c", patterns[i].tokens[j], j + 1 == patterns[i].tokens.size() ? '\n' : ' ');
            }
        }
    }
    fclose(out);
}

template<class T>
void dumpRankingList_labels(const string& filename, vector<pair<T, PATTERN_ID_TYPE>> &order, const double& threshold)
    {
        FILE* out = tryOpen(filename, "w");
        sort(order.rbegin(), order.rend());
        for (size_t iter = 0; iter < order.size(); ++ iter) {
            PATTERN_ID_TYPE i = order[iter].second;
            if (patterns[i].quality > threshold){
                fprintf(out, "%u\t", i);
                fprintf(out, "%u\t", patterns[i].label);
                fprintf(out, "%.10f\t", patterns[i].quality);
                for (int j = 0; j < patterns[i].tokens.size(); ++ j) {
                    fprintf(out, "%d%c", patterns[i].tokens[j], j + 1 == patterns[i].tokens.size() ? '\n' : ' ');
                }
            }
        }
        fclose(out);
    }



void dumpPatternStruct(){
    FILE *outfile;

    // open file for writing
    outfile = fopen ("tmp/patterns_struct.dat", "w");
    if (outfile == NULL)
    {
        fprintf(stderr, "\nError opend file\n");
        exit (1);
    }
    fwrite (&patterns, sizeof(patterns), 1, outfile);

    // close file
    fclose (outfile);

}


void dumpResults(const string& prefix)
{
    vector<pair<double, PATTERN_ID_TYPE>> order;
    for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++ i) {
        if (patterns[i].size() > 1 && patterns[i].currentFreq > 0) {
            order.push_back(make_pair(patterns[i].quality, i));
        }
    }
    dumpRankingList(prefix + "_multi-words.txt", order);

    order.clear();
    for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++ i) {
        if (patterns[i].size() == 1 && patterns[i].currentFreq > 0 && unigrams[patterns[i].tokens[0]] >= MIN_SUP) {
            order.push_back(make_pair(patterns[i].quality, i));
        }
    }
    dumpRankingList(prefix + "_unigrams.txt", order);

    order.clear();
    for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++ i) {
        if (patterns[i].size() > 1 && patterns[i].currentFreq > 0 || patterns[i].size() == 1 && patterns[i].currentFreq > 0 && unigrams[patterns[i].tokens[0]] >= MIN_SUP) {
            order.push_back(make_pair(patterns[i].quality, i));
        }
    }
    dumpRankingList(prefix + "_salient.txt", order);
}

void updatePatternsActiveLearning(){
        cout << "Reading labeled intermedia file from disk" << endl;
        FILE *infile;
        patterns.clear();
        // Open person.dat for reading
        infile = fopen ("tmp/patterns_struct.dat", "r");
        if (infile == NULL)
        {
            fprintf(stderr, "\nError opening file\n");
            exit (1);
        }
        // read file contents till end of file
        fread(&patterns, sizeof(patterns), 1, infile);
        cout << "Reading " << sizeof(patterns) << " patterns from disk" << endl;
        // close file
        fclose (infile);
    }

}

void loadLabelPatterns(const string& filename){

    FILE* in = tryOpen(filename, "r");
    cout << "AL rectifying ... " << endl;

    while (getLine(in)) {
        vector<string> tokens = splitBy(line, '\t');
        if (tokens.size() == 2) {
            TOKEN_ID_TYPE id;
            int label;
            fromString(tokens[0], id);
            fromString(tokens[1], label);

            patterns[id].label = label;


        }
    }

    }


void calculateTopAvgScore(){

//    calculate three scores: top 5% average, % with score > 0.9, percentage of middle-zone scores

    vector<pair<double, PATTERN_ID_TYPE>> order;
    int high_score_patterns=0;
    int middle_score_patterns=0;

    for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++ i) {
        if (patterns[i].size() > 1 && patterns[i].currentFreq > 0) {
            order.push_back(make_pair(patterns[i].quality, i));
            if ((double) patterns[i].quality > 0.9){
                high_score_patterns += 1;
            }else if(patterns[i].quality >0.6 && patterns[i].quality<0.8){
                middle_score_patterns += 1;
            }
        }
    }

    sort(order.rbegin(), order.rend());
    int i = 0;
    int size = order.size();
    double sum = 0;

    while(i < 0.05 * size){
        PATTERN_ID_TYPE idx = order[i].second;
        sum += patterns[idx].quality;
        i += 1;
    }

    // saving logs
    FILE* out = tryOpen("tmp/al_log.txt", "a");

    std::time_t result = std::time(nullptr);

    fprintf(out, std::asctime(std::localtime(&result)));
    fprintf(out, "\n");
    fprintf(out, "top5 percent average score is: %.10f\n", (double) sum/(i+1));
    fprintf(out, "% with score>0.9 is: %.10f\n", (double) high_score_patterns/size);
    fprintf(out, "% with 0.8 > score > 0.7 is: %.10f\n", (double ) middle_score_patterns/size);
    fprintf(out, "\n");

//    cout << "top5% average score is: " << sum/(i+1) << endl;
//    cout << "% with score>0.9 is: " << (double) high_score_patterns/size << endl;
//    cout << "% with 0.8 > score > 0.7 is: " << (double ) middle_score_patterns/size << endl;

    fclose(out);
}

#endif
