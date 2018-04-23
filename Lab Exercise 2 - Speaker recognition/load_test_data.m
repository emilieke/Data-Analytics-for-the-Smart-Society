%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Read test speech files
% Input:
%     - list_name: list name
%
% Output:
%    - x: vector with the test speech samples of all speakers
%    - y: vector with the test speaker ids
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x,y] = load_test_data(list_name, i)

% Read the list of the test speech files
fid = fopen(list_name);

if fid < 0
    fprintf('File %s does not exist\n', list_name);
    exit
end
info_speech = textscan(fid, '%s%f');
clase = int16(info_speech{2});    % speaker id of each file
fclose(fid);

% Extract the test speech samples of the specific
% "speaker_id" speaker
x = [];
y = [];
fname = info_speech{1}{i};
[aux fsaux] = audioread(fname);
id = clase(i);
x = [x; aux];
y = [y; id];



