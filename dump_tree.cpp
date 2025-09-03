void output_vector(std::ofstream& file,std::vector<double>* vec)
{
    if (vec->size()>0)
        file << "," << vec->at(0);
    else
        file << "," << -1;
}
void output_vector(std::ofstream& file,std::vector<int>* vec)
{
    if (vec->size()>0)
        file << "," << vec->at(0);
    else
        file << "," << -1;
}

void dump_tree()
{
    ofstream outputTrainingFile("mc17606_mu_positive_training.txt");
    ofstream outputValidationFile("mc17606_mu_positive_validation.txt");

    outputTrainingFile << "Event,MuonDecay,CHVL_Hit,CHVR_Hit,CHAMBER_Hit,BHC_Hit,BHD_Hit,BHC_CopyID,BHD_CopyID,BHC_Edep,BHD_Edep,SPSLF_Hit,SPSLR_Hit,SPSRF_Hit,SPSRR_Hit,SPSRF_CopyID,SPSRR_CopyID,SPSLF_CopyID,SPSLR_CopyID,SPSRF_Edep,SPSRR_Edep,SPSLF_Edep,SPSLR_Edep\n";
    outputValidationFile << "Event,MuonDecay,CHVL_Hit,CHVR_Hit,CHAMBER_Hit,BHC_Hit,BHD_Hit,BHC_CopyID,BHD_CopyID,BHC_Edep,BHD_Edep,SPSLF_Hit,SPSLR_Hit,SPSRF_Hit,SPSRR_Hit,SPSRF_CopyID,SPSRR_CopyID,SPSLF_CopyID,SPSLR_CopyID,SPSRF_Edep,SPSRR_Edep,SPSLF_Edep,SPSLR_Edep\n";

    int events = 0;
    for(int ind = 1; ind < 29; ind++)
    {
        std::cout << "Processing file: " << ind << std::endl;
        TString file = "root_files/mc17606_mu_positive_210MeV_LH2_15Apr25_" + TString::Format("%d.root",ind);
        TFile *f = new TFile(file);
        TTree *t = (TTree*)f->Get("T");

        int CHVL_Hit,CHVR_Hit,CHAMBER_Hit,BHC_Hit,BHD_Hit,SPSRF_Hit,SPSRR_Hit,SPSLF_Hit,SPSLR_Hit;
        bool MuonDecay;
        std::vector<int> *BHC_CopyID = nullptr;
        std::vector<int> *BHD_CopyID = nullptr;
        std::vector<double> *BHC_Edep = nullptr;
        std::vector<double> *BHD_Edep = nullptr;
        std::vector<double> *SPSRF_Edep = nullptr;
        std::vector<double> *SPSRR_Edep = nullptr;
        std::vector<double> *SPSLF_Edep = nullptr;
        std::vector<double> *SPSLR_Edep = nullptr;

        std::vector<double> *SPSRF_CopyID = nullptr;
        std::vector<double> *SPSRR_CopyID = nullptr;
        std::vector<double> *SPSLF_CopyID = nullptr;
        std::vector<double> *SPSLR_CopyID = nullptr;

        t->SetBranchAddress("MuonDecay",&MuonDecay);
        t->SetBranchAddress("CHVL_Hit",&CHVL_Hit);
        t->SetBranchAddress("CHVR_Hit",&CHVR_Hit);
        t->SetBranchAddress("CHAMBER_Hit",&CHAMBER_Hit);
        t->SetBranchAddress("BHC_Hit",&BHC_Hit);
        t->SetBranchAddress("BHD_Hit",&BHD_Hit);
        t->SetBranchAddress("SPSRF_Hit",&SPSRF_Hit);
        t->SetBranchAddress("SPSRR_Hit",&SPSRR_Hit);
        t->SetBranchAddress("SPSLF_Hit",&SPSLF_Hit);
        t->SetBranchAddress("SPSLR_Hit",&SPSLR_Hit);
        t->SetBranchAddress("BHC_CopyID",&BHC_CopyID);
        t->SetBranchAddress("BHD_CopyID",&BHD_CopyID);
        t->SetBranchAddress("SPSRF_CopyID",&SPSRF_CopyID);
        t->SetBranchAddress("SPSRR_CopyID",&SPSRR_CopyID);
        t->SetBranchAddress("SPSLF_CopyID",&SPSLF_CopyID);
        t->SetBranchAddress("SPSLR_CopyID",&SPSLR_CopyID);
        t->SetBranchAddress("BHC_Edep",&BHC_Edep);
        t->SetBranchAddress("BHD_Edep",&BHD_Edep);

        t->SetBranchAddress("SPSRF_Edep",&SPSRF_Edep);
        t->SetBranchAddress("SPSRR_Edep",&SPSRR_Edep);
        t->SetBranchAddress("SPSLF_Edep",&SPSLF_Edep);
        t->SetBranchAddress("SPSLR_Edep",&SPSLR_Edep);

        Long64_t nentries = t->GetEntries();
        for (Long64_t i = 0; i < nentries; i++) 
        {
            t->GetEntry(i);
            if(ind < 20)
            {
                outputTrainingFile << events << "," << MuonDecay << "," << CHVL_Hit << "," << CHVR_Hit << "," << CHAMBER_Hit << "," << BHC_Hit << "," << BHD_Hit;
                output_vector(outputTrainingFile,BHC_CopyID);
                output_vector(outputTrainingFile,BHD_CopyID);
                output_vector(outputTrainingFile,BHC_Edep);
                output_vector(outputTrainingFile,BHD_Edep);
                outputTrainingFile << "," << SPSLF_Hit << "," << SPSLR_Hit << "," << SPSRF_Hit << "," << SPSRR_Hit;
                output_vector(outputTrainingFile,SPSRF_CopyID);
                output_vector(outputTrainingFile,SPSRR_CopyID);
                output_vector(outputTrainingFile,SPSLF_CopyID);
                output_vector(outputTrainingFile,SPSLR_CopyID);
                output_vector(outputTrainingFile,SPSRF_Edep);
                output_vector(outputTrainingFile,SPSRR_Edep);
                output_vector(outputTrainingFile,SPSLF_Edep);
                output_vector(outputTrainingFile,SPSLR_Edep);
                outputTrainingFile << "\n";
            }
            else
            {
                outputValidationFile << events << "," << MuonDecay << "," << CHVL_Hit << "," << CHVR_Hit << "," << CHAMBER_Hit << "," << BHC_Hit << "," << BHD_Hit;
                output_vector(outputValidationFile,BHC_CopyID);
                output_vector(outputValidationFile,BHD_CopyID);
                output_vector(outputValidationFile,BHC_Edep);
                output_vector(outputValidationFile,BHD_Edep);
                outputValidationFile << "," << SPSLF_Hit << "," << SPSLR_Hit << "," << SPSRF_Hit << "," << SPSRR_Hit;
                output_vector(outputValidationFile,SPSRF_CopyID);
                output_vector(outputValidationFile,SPSRR_CopyID);
                output_vector(outputValidationFile,SPSLF_CopyID);
                output_vector(outputValidationFile,SPSLR_CopyID);
                output_vector(outputValidationFile,SPSRF_Edep);
                output_vector(outputValidationFile,SPSRR_Edep);
                output_vector(outputValidationFile,SPSLF_Edep);
                output_vector(outputValidationFile,SPSLR_Edep);
                outputValidationFile << "\n";
            }
            events++;
        }
        f->Close();
    }
    outputTrainingFile.close();
    outputValidationFile.close();

    return;
}