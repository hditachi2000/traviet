#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath> 
#include <cstdlib>
#include <random> 
#include <algorithm> 
#include <random>
#include <ctime> 

using namespace std;

//#define GroupNumber 4
//define min_split 2
//define max_depth 10
// #define trainRatio 0.7 


class Sample {
public:
    vector<float> features;           // tập chứa  giá trị của các feature tương ứng với sample đó 
    int label;                     // nhãn của sample 
};

// Define a class to represent a single node in the decision tree
class Node {
public:
    int feature;                      // biến này chứa id của Feature dùng để chia nhánh 
    float branchValue;               // giá trị nút cha chia nhánh xuống nút con hiện tại 
    int label;                      // nhãn đại diện cho node đó, nếu node đó là node chứa các lá
    int numSample;                 // đếm số lá thuộc node, biến này dùng cho siêu tham số min_split 
    vector<Node*> children;       // vector chứa các node con 
    bool is_leaf=false;          // xác nhận có phải node chứa toàn lá không 
    float ac;                   // độ chắc chắn tại node
};

class Data{
public:
    unordered_map<int,Sample> AllSample;      // tập các giá trị feature của 1 sample và id của sample đó 
    unordered_map<int,float> feature;              // tập id và tỉ trọng tương ứng của feature 
    unordered_set<int> constantFeature;          // tập id của những feature đã cố định ở dạng catogory -> không thay đổi khi huấn luyện 
};                                          //

void printData(Data a){
    cout << "Num Sample: " << a.AllSample.size() << endl;
    for(auto sample : a.AllSample){
        cout << "ID " << sample.first <<": " ;
        for(auto idf : sample.second.features){
            cout << idf << "   \t";
        }
        cout << endl;
    }
    cout << "Label : " << endl;
    for(auto sample : a.AllSample){
            cout << sample.second.label << "- " ;
        }
}

// Hàm tính entropy của 1 tập sample 
float entropy(unordered_map<int,Sample> samples) {
    unordered_set<int> labels; 
    for (auto sample : samples) {
        labels.insert(sample.second.label);
    }
    float result = 0;
    for (auto label : labels) {
        float p = 0;
        for (auto sample : samples) {
            if (sample.second.label == label) {
                p += 1;
            }
        }
        p /= samples.size();
        result -= p * log2(p);
    }
    return result;
}

// hàm tính Information Gain của 1 feature 
float infoGain(unordered_map<int, Sample> Samples, int idFeature) {
    float baseEntropy = entropy(Samples);
    unordered_set<float> featureValues;          // tập các giá trị nhóm của feature 
    for (auto sample : Samples) {
        featureValues.insert(sample.second.features[idFeature-1]);
    }
    float result = 0;
    for (auto value : featureValues) {
        //cout << "value: " << value << " - " ;
        unordered_map<int,Sample> subset;             // tạo tập con chứa id các sample cùng 1 giá trị 
        for (auto sample : Samples) {
            if (sample.second.features[idFeature-1] == value) {
                //cout << "sample.first: " << sample.first << " - " ;
                subset[sample.first] = sample.second;         // thêm id vào tập con 
                //cout<< endl;
            }
        }
        float p = (float)subset.size() / Samples.size();
        result += p * entropy(subset);
    }
    return baseEntropy - result;
}

// chuẩn hóa về dạng category - nhóm 

/* Hàm chạy được với cả những data hoặc feature đã ở sẵn dạng category . 1 unordered_set sẽ lưu lại ID của những feature này, và số nhóm của chúng
  sẽ không bị thay đổi trong quá trình trainning 
*/
// 1 map chứa số nhóm của các feature được coi như 1 hyperparametter, sẽ được cài đặt và huấn luyện về sau 
void Standard(Data& a,unordered_map<int,int> featureGroup){
    // thăm dò : xem dữ liệu đã ở dạng category hay chưa 
  
    for(auto& idf : a.feature){    // duyệt qua tất cả id của các feature 
        float minValue = 999999.0;
        float maxValue = -99999.0;
        unordered_set<float> featureValue;
        for(auto sample : a.AllSample){                   // duyệt 1 cột feature
            //cout << sample.second.features[idf-1] << " ";
            if(sample.second.features[idf.first-1] < minValue){
                minValue = sample.second.features[idf.first-1];
            }
            if(sample.second.features[idf.first-1] > maxValue){
                maxValue = sample.second.features[idf.first-1];
            }
            featureValue.insert(sample.second.features[idf.first-1]);
        }
        //cout << endl;
        //cout <<"Min: " << minValue << " - Max: " << maxValue << endl;
        if(featureValue.size() < 10){    // nếu feature có < 10 gtri khác nhau -> category -> chuyển sang feature tiếp theo
            a.constantFeature.insert(idf.first);
            continue;
        }

        float range = maxValue - minValue;                    // khoảng cách giữa max và min
        float average = range / featureGroup[idf.first];     // độ rộng của 1 thùng 

        // đưa tất cả feature trong cột hiện tại về category
        for(auto& sample : a.AllSample){
            //cout << " sample.second.features[idf-1] : " << sample.second.features[idf-1] << " - ";
            //cout << "minvalue : " << minValue << endl;
            int a = (sample.second.features[idf.first-1] - minValue ) / average;
            sample.second.features[idf.first-1] = minValue + a*average;
            //cout << sample.second.features[idf-1] << " - ";
        }
    }
}

// Load dữ liệu từ file vào Data 
void LoadData(Data& a, string filename){
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file." << endl;
        return;
    }

    string line;
    getline(file, line);
    int countSample = 1;                      // số sample  
    while (getline(file, line)) {
        istringstream ss(line);
        Sample newS;
        string f;
        int countFeature = 1;                  
        while (getline(ss, f, ',')) {
            //if (countFeature > 10) break;   // giới hạn số feature , bỏ qua dòng này để lấy tất cả feature 
            float newF;
            try {
                newF = stof(f);
            } catch (const exception& e) {
                cerr << "Failed to convert to float: " << f << endl;
                break;
            }
            if(countFeature == 1){               //  cột số 1 là nhãn - có thể điều chỉnh tại vị trí bất kỳ cho các file khác nhau 
                if(newF>=1){
                    newS.label = 1;
                }
                else newS.label = 0;
            }else{
                newS.features.push_back(newF);
                a.feature[countFeature-1] = 1;               //thêm id của feature với tỉ trọng mặc định là 1 
                //cout << "c: " << c-1 << " @ ";
            }
            countFeature++;
        }
        //if (countSample > 200) break;                  // giới hạn số sample , bỏ qua dòng này để lấy tất cả sample
        
        a.AllSample[countSample] = newS;   // gán id cho sample
        countSample++;
        countFeature = 1;
    }
    file.close();
}

pair<Data, Data> splitData(const Data& a, float trainRatio){
    // số sample lấy theo Ratio : tỉ lệ giữa train set và data 
    int trainSize = static_cast<int>(a.AllSample.size() * trainRatio);
    int testSize = a.AllSample.size() - trainSize;

    // lưu ID của tất cả sample 
    vector<int> sampleIds(a.AllSample.size());
    int index = 0;
    for (const auto& sample : a.AllSample) {
        sampleIds[index++] = sample.first;
    }

    // lấy ngẫu nhiên sample từ data 
    random_device rd;
    mt19937 generator(rd());
    shuffle(sampleIds.begin(), sampleIds.end(), generator);

    // tạo tập train 
    Data train_Data;
    for (int i = 0; i < trainSize; ++i) {
        int sampleId = sampleIds[i];
        train_Data.AllSample[sampleId] = a.AllSample.at(sampleId);
        //cout << "A label : " << a.AllSample.at(sampleId).label <<"- Train label: " << train_Data.AllSample[sampleId].label << endl;
    }
    train_Data.feature = a.feature;

    // tạo tập test , dựa trên số sample còn lại 
    Data test_Data;
    for (int i = trainSize; i < trainSize + testSize; ++i) {
        int sampleId = sampleIds[i];
        test_Data.AllSample[sampleId] = a.AllSample.at(sampleId);
    }
    test_Data.feature = a.feature;

    // Return the train_Data and test_Data
    return make_pair(train_Data, test_Data);
}

Node* buildTree(Data& a,int min_split,int max_depth) {
    int depth = 0;
    Node* node = new Node();
    node->numSample = a.AllSample.size();
    unordered_set<int> lab;            // Tập các nhãn  
    for (auto sample : a.AllSample) {
        lab.insert(sample.second.label);
    }
    
    // tất cả cùng nhãn , gán node là lá 
    if (lab.size() == 1) {
        //cout << "here 1! " << endl;
        node->is_leaf = true;
        node->label = *lab.begin();
        node->ac = 1.0;
       // cout << "Leaf Accuracy: " << node->ac <<"\t" << " - num Leaf: " << a.AllSample.size() << endl;
        return node;
    }

    // Còn quá ít sample để chia    || ko còn feature    || độ sâu đạt tối đa 
    if (node->numSample < min_split || a.feature.empty() || depth >= max_depth) {
        //cout << "here 2 ! " << endl;
        int mostLab, max = -999;
        unordered_map<int, int> labelCount;
        for (auto sample : a.AllSample) {
            labelCount[sample.second.label]++;
            if (labelCount[sample.second.label] > max) {              // tìm nhãn phổ biến nhất 
                max = labelCount[sample.second.label];                 
                mostLab = sample.second.label;
            }
        }
        // cout <<"Most label: " << node->label <<  endl;
        node->label = mostLab;
        node->is_leaf = true;
        node->ac = (float)max / a.AllSample.size();           // độ chắc chắn của lá 
        //cout << "Leaf Accuracy: " << node->ac <<"\t" << " - num Leaf: " << a.AllSample.size() << endl;
        return node;
    }

    //
    double maxIG = -1;
    for (auto feature : a.feature) {
        double IG = infoGain(a.AllSample, feature.first) * feature.second;    // nhân với tỉ trọng của feature 
        if (IG > maxIG) {
            maxIG = IG;
            node->feature = feature.first;
            //cout <<"ID " <<feature.first<< " - " <<  "IG: " << IG << endl; 
        }
    }
    //cout << "MaxIG: " << maxIG << " - ";
    //cout << "Feature : " << node->feature << endl;
    a.feature.erase(node->feature);

    // PHÂN NHÓM VÀ ĐỆ QUY 
    unordered_set<float> featureValue;           // tạo set chứa các giá trị của feature được chọn 
    for (auto sample : a.AllSample) {
        featureValue.insert(sample.second.features[node->feature - 1]);          // duyệt qua 1 cột 
    }
    //cout << sample.second.features[node->feature - 1] << " - " ;
    for (auto value : featureValue) {                     
        //cout << "here 4 : tao nhom de quy" << endl;
        //cout << "Value : " << value << endl;
        Data subData = Data();                     // Data con với các sample và feature mới cho vòng đệ quy tiếp theo 
        for (auto sample : a.AllSample) {
            if (sample.second.features[node->feature - 1] == value){
                subData.AllSample[sample.first] = sample.second;     // gán các sample
                subData.feature = a.feature;                   //  gán tập feature 
            }
        }
        //cout << "SubData size: " << subData.feature.size() << endl;
        if (subData.AllSample.empty()) {                // không còn sample để chia 
            //cout << "here 5 : het value feature de chia" << endl;
            int mostLab, max = -999;
            unordered_map<int, int> labelCount;
            for (auto sample : a.AllSample) {
                labelCount[sample.second.label]++;
                if (labelCount[sample.second.label] > max) {
                    max = labelCount[sample.second.label];
                    mostLab = sample.second.label;
                }
            }
            Node* childNode = new Node();
            childNode->label = mostLab;             // gán nhãn phổ biến
            childNode->is_leaf = true;              // gán là lá
            childNode->ac = (float)max / a.AllSample.size();          
            childNode->branchValue = value;                 // Gán giá trị phân nhánh cho nút con 
            node->children.push_back(childNode);
        }
        else {
            //cout << "here 6: de quy voi node con" << endl;
            Node* childNode = buildTree(subData,min_split,max_depth); 
            max_depth--;                                 // tăng độ sâu của cây lên 1 
            childNode->branchValue = value;                 // gán giá trị phân nhánh cho nút con
            node->children.push_back(childNode);
        }
    }
    return node;
}

bool predictSample(Node* node, Sample s) {          // dự đoán đúng/sai cho 1 sample 
    if (node->is_leaf) {
        return ( node->label == s.label);
    }
    
    for (auto child : node->children) {         // tìm nhánh rẽ 
        if (child->branchValue == s.features[node->feature - 1]) {     // khớp giá trị chia nhánh 
            return predictSample(child, s); 
        }
    }
    return false;
}

int predictClassSample(Node* rootNode, const Sample& s) {
    while (!rootNode->is_leaf) {
        int featureIndex = rootNode->feature - 1;
        float sampleValue = s.features[featureIndex];
        
        bool foundChild = false;
        for (const auto& childNode : rootNode->children) {    // tìm nhánh rẽ 
            if (childNode->branchValue == sampleValue) {    // khớp giá trị chia nhánh 
                //cout << " childMode->branchValue: " << childNode->branchValue << endl;
                rootNode = childNode;
                foundChild = true;
                break;
            }
        }
        

        if (!foundChild) {
            return rootNode->label;
        }
    }
    
    return rootNode->label;
}

float predictData(Node* node, Data a){                 // trả về tỉ lệ dự đoán chính xác trên 1 data 
    int count1 =0 , count2=0;
    for(auto idf : a.AllSample){
        if(predictSample(node,idf.second)){
            count1++;
        }else{
            count2++;
        }
    }
    return (float)count1/(count1+count2);
}
//thêm 1 sample từ 1 data vào 1 data khác 
void addSample(Data& a, Data& sub, int id){
    sub.AllSample[id] = a.AllSample[id];
    //sub.feature = a.feature;
}

// nhân bản các sample có sẵn trong 1 data cho tới khi đạt được số lượng Sample = numSample
// hàm đã được tối ưu với độ phức tạp N+m:   N: Numsample, m : số sample thêm vào < N
void fillData(Data& a,int numSample){
    unordered_set<int> Yet;          // tạo tập ID đang có 
    //cout <<"YET ID: " << endl;
    for(auto id : a.AllSample){
        Yet.insert(id.first);
        //cout <<id.first <<" - " ;
    }
    //cout << endl;
    vector<int> notYet;   // tập id trong data gốc mà chưa có trong data con 
    //cout <<"NOT YET ID: " << endl;
    for(int id =1; id <= numSample;id++){
        if(Yet.find(id) == Yet.end()){
            notYet.push_back(id);
            //cout <<id << " - " ;
        }
    }
    //cout <<endl;
    while(a.AllSample.size() < numSample){
        for(auto id : Yet){
            int ID = notYet.back();
            notYet.pop_back();
            //Nhân bản sample :
            a.AllSample[ID] = a.AllSample[id];
            if(a.AllSample.size() == numSample) break;
        }
    }
    //cout <<"NUM FEATURE: " << a.feature.size() << endl;
    //printData(a);
    //cout <<"END FILL DATA" << endl;
}

void killFeature(Data& a, int featureID){
    a.feature.erase(featureID);                 // xóa feature tại Data
    for(auto& sample : a.AllSample){         // duyệt qua cột chứa id feature
        auto iter = sample.second.features.begin() + featureID-1;       
        sample.second.features.erase(iter);       // xóa feature tại sample 
    }

}
void deleteFeature(Data& a, int featureID) {
    a.feature.erase(featureID);
    for (auto& pair : a.AllSample) {
        vector<float>& features = pair.second.features;
        if (featureID >= 1 && featureID <= features.size()) {
            features.erase(features.begin() + featureID - 1);
        }
    }
}


 // Nhân bản Data (tương tự boostrap)
//CẬP NHẬT :  Các sub-Data hạn chế việc giao thoa feature và sample / Đảm bảo tính ngẫu nhiên 
/* Solve : 2 unordered_map "Selected" sẽ lưu lại các sample / feature đã được chọn . Quá trình chọn ngẫu nhiên sample/feature cho sub_Data 
  sẽ bỏ qua những sample/feature đã được chọn . Nếu tổng số sample/feature trong tất cả subdata bằng "n" lần số sample/feature tại Data gốc
  mỗi lần Selected đầy , tập này sẽ được clear lại. 
*/
vector<Data> ClonesData(Data a, int numSubData){
    vector<Data> subDataList;
    int numSample = a.AllSample.size();
    int numFeature = a.feature.size() ;        // lấy ra căn 2 số feature gốc 
    int numDeleteFeature = a.feature.size() - numFeature;

    // tạo ra tập id của các feature
    unordered_set<int> featureID;
    for(auto feature : a.feature){
        featureID.insert(feature.first);
    }
    // tạo mảng chứa id của các sample 
    int sampleID[numSample];
    int c=0;
    for(auto sample : a.AllSample){
        sampleID[c] = sample.first;
        //cout << sampleID[c] <<"-";
        c++;
    }
    //cout << endl;
    //printData(a);
    //cout << endl;
    // tạo ra các tập data con 
    for(int i=0; i<numSubData; i++){
        //cout <<"here" <<endl;
        Data subData;
        subData.feature = a.feature;               // sao chép bộ feature từ data gốc sang data con
        //int numSubSample = numSample;
        int numSubSample = rand()%numSample*0.4 + numSample*0.2;   // chọn ra 20% - 60% số sample từ tập gốc
        //cout <<"Num SubData Sample: " << numSubSample << endl;
        unordered_set<int> added;
        for(int k = numSubSample; k>0;k--){
            //int id = rand()%(a.AllSample.size()) + 1;
            int iter = rand()%numSample;     // chọn ra ngẫu nhiên id của sample trong tập sample 
            int id = sampleID[iter];
//
            //cout << "ID sample add: " << id << endl;
            if(added.find(id) != added.end()){    // nếu sample đã được chọn
                k++;
            }
            else{
                added.insert(id);
                subData.AllSample[id] = a.AllSample[id];
                //cout <<"SUB DATA ADDED: " <<endl;
                //printData(subData);
                //cout << endl;
                //cout << "ID ADDED : " << id << endl;
                //cout <<"AddSample: " << id << endl; 
            }
        }
        // tạo xong 1 sub data, với số feature = numSubSample 

        // đưa số feature về căn bậc 2 của tổng số feature  
        unordered_set<int> deleted;
        for(int k =1; k <= numDeleteFeature; k++){
            //cout << "here 10 !" << endl;
            int id = rand()%(a.feature.size()) + 1;
            //cout << endl;
            //cout <<"random id: " << id << endl;
            if(deleted.find(id) == deleted.end()){
                int count = 0;
                for(auto d : deleted){
                    if(id>d) count++;               // căn chỉnh con trỏ tới phạm vi chính xác khi vector bị co lại  
                }
                //cout << " kill id: " << id-count<<endl;
                //printData(subData);
                killFeature(subData,id-count);
                //deleteFeature(subData,id-count);
                deleted.insert(id);
                //cout <<"id in kill feature: " << id << endl;
            }else{
                k--;
            }
        }
        // Làm đầy dữ liệu để bằng số sample của dữ liệu gốc 
        fillData(subData,a.AllSample.size());
        subDataList.push_back(subData);
    }    
    return subDataList;
}

struct RandomForest{
    Data a;
    Data trainData;
    Data validation;
    Data testing;
    unordered_map<int,Node*> root;                          // vector lưu trữ các nút gốc  
    int maxDepth;
    int minSplit;
    int numFeature;
    int numTree;                                        // n_estimmor
    unordered_map<int,float> featureWeight;            // tỉ trọng các feature 
    unordered_map<int,int> featureGroup;              // số nhóm được chia trong mỗi feature

    RandomForest(){};
    RandomForest(string filename){
        LoadData(this->a,filename);
    }

    void ProcessingData(){
        this->maxDepth = 10; 
        this->minSplit = 3;
        this->numFeature = sqrt(a.feature.size());
        this->numTree = 10;
        for(int i=1; i<=this->a.feature.size();i++){
            featureWeight[i] = 1;                         // mặc định ban đầu tất cả tỉ trọng là 1 - bằng nhau 
            featureGroup[i] = 3;                         // mặc định số nhóm của mỗi feature là 4 - bằng nhau 
        }
        cout << endl;
        Standard(this->a,featureGroup);
        //cout <<" DATA GOC : " << endl;
        //printData(a);
        // Phân chia dữ liệu ra 3 tập trainData, Validation, testing 
        Data b;
        pair<Data,Data> train = splitData(a,0.8);        // 20% cho tập test
        b = train.first;
        testing = train.second;
        pair<Data,Data> valid = splitData(b,0.75);       // 60% cho tập train, 20 % cho tập validation 
        trainData = valid.first;
        validation = valid.second;
        // TỪ đây, các cây sẽ làm việc với dữ liệu con trong tập trainData
        // Trong quá trình huấn luyện và cắt tỉa, các cây sẽ làm việc với tập validation
    }

    void MakeForest(){
        //cout << "NUm tree: " << numTree << endl;
        vector<Data> DataList = ClonesData(trainData,numTree);    // tạo 1 tập dữ liệu con với số sub-data = số cây 
        //cout <<"Data list size: " << DataList.size() << endl;
        // Tạo rừng :
        for(int i = 0; i < numTree ; i++){
            Node* rootNode = buildTree(DataList[i],minSplit,maxDepth);
            //cout << "DATA " <<i<<": " <<endl;
            //printData(DataList[i]);
            this->root[i] = rootNode;                            // Chú ý : ID của các cây được đánh số từ 0, ko phải từ 1 
            //cout <<"Num tree: " << root.size() << endl;
        }
    }

    // hàm dự đoán kết quả của 1 sample trên quy mô toàn bộ rừng 
    pair<int,float> predClassSample(Sample s){
        int totalPredict = 0;
        unordered_map<int,int> predictClass;      // tập chứa kết quả các nhãn được dự đoán và số lượng dự đoán của nó 
        for(auto node : root){                 // duyệt qua tất cả các cây 
            int predict =  predictClassSample(node.second,s);
            if(predict != -1){       // dự đoán thành công 
                predictClass[predict]++;                             // tăng số lượng nhãn lên 1
                totalPredict++;
            }
        }
        int max = -1;
        int mostPredict;
        if(predictClass.size() == 0) return {-1,0};
        for(auto predict : predictClass){           // chọn nhãn dựa trên đa số
        //cout << "predict.second: " << predict.second<<endl;
            if(predict.second > max){
                max = predict.second;
                mostPredict = predict.first;
            }
        }
        //cout << "Max: " << max << endl;
        //cout << "Most Predict: " << mostPredict << endl;
        float certainly = max / totalPredict;           // độ chắc chắn của dự đoán 
        pair<int,float> result = {mostPredict,certainly};
        return result;    // trả về nhãn được nhiều cây ủng hộ nhất cùng với tỉ lệ ủng hộ
    }
    // độ chính xác dự đoán của rừng
    float predData(Data t){                // dự đoán trên 1 tập Data 
        float count1=0, count2=0;
        for(auto sample : t.AllSample){
            if(predClassSample(sample.second).first == -1) continue;
            if(predClassSample(sample.second).first == sample.second.label){
                count1++;
            }
            else count2++;
        }
        //cout <<"count 1: " << count1 << " - count 2: " << count2 << endl;
        //cout << "Accuracy of Forest in Data: " << count1/(count1+count2) << endl;
        return count1/(count1+count2);
    }
    // Hàm thực hiện lời giải và trả về độ chính xác của rừng trên tập validation
    float Solution(Data data){
        ProcessingData();
        MakeForest();
        return predData(data);
    }

    // Hàm huấn luyện, trả về tập node tối ưu nhất || Mỗi lần huấn luyện sẽ tạo 1 rừng cây mới 
    // các hàm con đã được test và cài đặt tham số phục vụ cho quá trình huấn luyện 
    // C1 : Training theo lưới/grid search 
    /*C2 : Cập nhật lưới nâng cấp : Theo Accuracy / overfiting :
    - Accuracy giảm : tăng numTree, tăng featureGroup 
    - Overfitting = Accuracy(training) - Accuracy(validation)
        Overfitting tăng : giảm minSplit, giảm maxDepth /  ngược lại
    - Đối với riêng featureWeight : thăm dò : ban đầu, tăng/giảm trọng số tại từng feature, để phỏng đoán feature nào ảnh hưởng như thế nào tới Accuracy và
                                              overfitting . Sau đó mỗi phần tử trong featureWeight sẽ được coi như 1 hyperparameter trong quá trình traning 
     Một unprdered_map ứng với các hyperparameter và xu hướng ảnh hưởng của chúng lên accuracy và overfitting . Quá trình tranning sẽ dựa vào xu hướng thay đổi
     của accuray và overfitting để điều chỉnh các hyperparameter cho phù hợp . 
    */
    unordered_map<int,Node*>  Trainning(int epochs){
        float bestAcc = Solution(validation);        // Tạo lời giải và kết quả ban đầu 
        unordered_map<int,Node*> bestRoot = this->root;
        float PreAcc;
        for(int i = 0; i<epochs; i++){
            // số lượng sample và feature để tạo 1 sub-data tạm không thể thay đổi / chưa được cài đặt làm tham số 

            // cập nhật minSplit
            // cập nhật maxDepth
            // cập nhật featureWeight : a.features.second 
            // cập nhật featureGroup/số nhóm trong 1 feature : hạn chế . CHỉ tăng khi độ chính xác kém / bế tắc (2^n)
            // cập nhật numTree / số cây con :: càng tăng càng tốt, nhưng gây nặng và chậm 

            float acc = Solution(validation);
            cout << "New Accuracy: " << acc << endl;
        }
    }
    
    // Cắt tỉa trên quy mô toàn bộ rừng 
    void Prunning(){
        // C1: Loại bỏ cây có accuray nhỏ hơn rừng trên tập validation 
        float MasterAcc = predData(validation);      //
        cout << "Master Accuracy: " << MasterAcc <<endl;
        for(int id = 0; id<root.size();id++){            
            float acc = predictData(root[id],validation);        // độ chính xác của cây 
            cout << "Accuracy of sub-tre: " << acc << endl;
            if(acc < MasterAcc){                        // độ chính xác nhỏ hơn rừng
                Node* node = root[id];
                root.erase(id);             // xóa node gốc của cây khỏi rừng 
                float forestAccuracy = predData(validation);
                if(forestAccuracy <= MasterAcc){                 // Nếu không cải thiện
                    root[id] = node;                    // thêm node vào lại 
                    cout << "New Forest Accuracy : " << forestAccuracy << endl;
                    cout << " Back Step !" << endl;
                }
                else {
                    MasterAcc = forestAccuracy;               // cải thiện, cập nhật tỉ lệ cao nhất 
                    cout << "New Master Accuracy: " << MasterAcc << endl; 
                }
            }
        }

        // C2:  Các cây với các nhánh có accuracy = 1/numClass , sẽ được thử thay đổi nhãn thành class khác 
    }

};

// CẮT TỈA / PRUNNING : thông thường, 1 cây quyết định lớn sẽ được cắt tỉa để làm giảm kích thước, overfitting và tăng tốc độ của cây
// Tuy nhiên , 1 rừng cây được tạo từ những cây riêng lẻ, có kích thước nhỏ hơn nhiều . Việc cắt tỉa được thực hiện trên quy mô toàn bộ rừng cây 

/*Cắt tỉa cấp độ Toàn bộ rừng :
1. Đánh giá từng cây bằng Lỗi ngoài túi - Dữ liệu bị bỏ lại trong quá trình nhân bản dữ liệu / cách này ko dùng do data con ko được đánh số 
2. Sử dụng validation_set : Đánh giá lỗi từng cây 
3. Loại bỏ cây xấu 
*/ 


int main() {
    srand(time(0));
    string filename = "Cancer_Data.csv";

    RandomForest forest = RandomForest(filename);
    forest.ProcessingData();
    forest.MakeForest();
    int count1=0, count2=0;
    /*
    for(auto node : forest.root){
        cout << "TRAIN ACCURACY: " << predictData(node.second,forest.testing) << endl;
        for(auto sample : forest.testing.AllSample){
            cout << "predict : " << predictClassSample(node.second,sample.second) << " - label : " << sample.second.label <<endl;
            if(predictClassSample(node.second,sample.second) == sample.second.label) count1++;
            count2++;
        }
        break;
        cout <<"ACCURACY: " << (float) count1/count2 << endl;
    }
    cout <<"ACCURACY: " << (float) count1/count2 << endl;
    */
    cout << "Accuracy in testing set: " << forest.predData(forest.testing);
    forest.Prunning();

   /*
    Data a = Data();
    LoadData(a,filename); 
    unordered_map<int,int> featureGroup;  // = {{1,4},{2,4},{3,4},{4,4},{5,4},{6,4},{7,4},{8,4},{9,4},{10,4},{11,4},{12,4},{13,4},{15,4}};
    for(auto id : a.feature){
        featureGroup[id.first] = 4;    // số nhóm của 1 feature 
    }
    Standard(a,featureGroup);
    //printData(a);
    pair<Data, Data> dataSplit = splitData(a,0.8);
    Data train_Data = dataSplit.first;
    Data test_Data = dataSplit.second;
    pair<Data,Data> s = splitData(train_Data,0.75);
    Data train = s.first;
    //cout <<"TRAIN DATA: " <<endl;
    //printData(train);
    //fillData(a,30);
    //killFeature(a,2);
    //deleteFeature(a,1);

    //vector<Data> DataList;
    //DataList = ClonesData(train,100);
    //cout <<"DATALIST SIZE: " <<DataList.size() << endl;
    //for(auto data : DataList){
        //cout <<"NEW DATA" << endl;
        //printData(data);
    //}
    //pair<Data, Data> dataSplit = splitData(a,0.6);
    //Data train_Data = dataSplit.first;
    //Data test_Data = dataSplit.second;
    //cout << "Train size: " << train_Data.AllSample.size() << endl;
    //printData(train_Data);
    //cout << "test size: " << test_Data.AllSample.size() << endl; 
    //printData(test_Data);
    //cout <<"DATA GOC: ----------------------" << endl;
    //printData(a);
    //cout <<"Entropy: "<< entropy(a.AllSample) << endl;
    //cout <<"IG: " << infoGain(a.AllSample,2)<<endl;
    Node* rootNode = buildTree(train_Data,3,10);
    /*
    for(auto data : DataList){
        //cout << "DATA CON: -----------------" << endl;
        //printData(data);
        Node* rootNode = buildTree(data,3,10);
        cout << endl;
        cout << " TRAIN ACCURACY : " << predictData(rootNode,data);
        //cout << " TEST ACCURACY : " << predictData(rootNode,test_Data);
        cout << endl;
    }
    
    //cout << endl;
    cout << " TRAIN ACCURACY : " << predictData(rootNode,train_Data);
    cout << " TEST ACCURACY : " << predictData(rootNode,test_Data);
    //RandomForest f1 = RandomForest(filename);
    */
    return 0;
}
