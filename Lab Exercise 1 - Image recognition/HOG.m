% Histogram of Oriented Gradients (HOG)
%
% Function: 
% H=HOG(block, nwin_x, nwin_y, num_or)
% 
% Inputs
%   -block: is the block from with the HOG is extracted. The size of each 
%    block will be reduced to (m-2*margen)x(n+2*margen), where m is the number
%    of and n is the number of columns. In our experiments mxn=16x16.
%
%   -x_cellnum: Number of spatial cells in x-direction. In our experiments, nwin_x=2;
%
%   -y_cellnum: Number of spatial cells in x-direction. In our experiments, nwin_y=2;
%
%   -num_grad_or: Number of gradient orientations. In our experiments, num_or=8;
%   
% Outputs
%   -H: HoG descriptor associated with Im. It is a vector of lenght nwin_x*nwin_y*num_or; 
%

function H=HOG(block, x_cellnum, y_cellnum,num_grad_or)
% Sobel masks to compute spatial gradients
hx=[1 0 -1; 2 0 -2; 1 0 -1]/2;
hy= [1 2 1;0 0 0; -1 -2 -1]/2;

% Gradient computation

grad_xr = -conv2(double(block),hx, 'same'); 
grad_yu = -conv2(double(block),hy, 'same'); 
% grad_xr = -imfilter(double(Im),hx, 'same'); 
% grad_yu = -imfilter(double(Im),hy, 'same'); 

% margin: we discard the gradients of the patch border (to avoid
% significant artificial gradients.  In our experiments, margin=1;
% discard the gradients of the patch border 
margin=1;
grad_xr=grad_xr(margin+1:end-margin,margin+1:end-margin);
grad_yu=grad_yu(margin+1:end-margin,margin+1:end-margin);

% Computing magnitud and orientations
% Orientation is computed in [-pi,pi]
angles=atan2(grad_yu,grad_xr); 
magnit=((grad_yu.^2)+(grad_xr.^2)).^.5;

% Reduced dimensions of the patch
[L,C]=size(grad_xr);

% Initialization of the HOG descriptor
H=zeros(x_cellnum*y_cellnum*num_grad_or,1); 

% Steps to move from one cell to another 
step_x=floor(C/(x_cellnum+1));
step_y=floor(L/(y_cellnum+1));

cont=0;
% Offset for histogram 
offset=num_grad_or/2+1;


for n=0:y_cellnum-1
    for m=0:x_cellnum-1
        cont=cont+1;
        
        angles2=angles(n*step_y+1:(n+2)*step_y,m*step_x+1:(m+2)*step_x);
        magnit2=magnit(n*step_y+1:(n+2)*step_y,m*step_x+1:(m+2)*step_x);
        
        % transformation into a vecto
        v_angles=angles2(:);
        v_magnit=magnit2(:);
        % Number of pixels
        K=max(size(v_angles));

        % Histogram
        H2=zeros(num_grad_or,1);
        salto=2*pi/num_grad_or;
        
        for k=1:1:K
            bin=floor(v_angles(k)/salto)+offset;
            if bin>8
                bin=8;
            end
            H2(bin)=H2(bin)+v_magnit(k);
        end
        H((cont-1)*num_grad_or+1:cont*num_grad_or,1)=H2;

    end
end
% L2-normalization 
H=H/(norm(H)+1e-10);
% Sum to one
H=H/(sum(H)+1e-10); 
