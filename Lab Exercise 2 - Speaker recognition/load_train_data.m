%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Read training speech files from a particular speaker
% Input:
%     - list_name: list name
%     - speaker_id: speaker identification number
%
% Output:
%    - x: vector with the training speech samples of the
%         "speaker_id" speaker
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x] = load_train_data(list_name, speaker_id)

% Read the list of the training speech files
fid = fopen(list_name);
if fid < 0
    fprintf('File %s does not exist\n', list_name);
    exit
end
info_speech = textscan(fid, '%s%f');
numfich = length(info_speech{1}); % total number of training files
clase = int16(info_speech{2});    % speaker id of each file
fclose(fid);

% Extract the training speech samples of the specific
% "speaker_id" speaker
x = [];
fich_spk = find(clase == speaker_id);
for i=1:length(fich_spk)
  fname = info_speech{1}{fich_spk(i)};
  [aux fsaux] = audioread(fname);
  x = [x; aux];
end


