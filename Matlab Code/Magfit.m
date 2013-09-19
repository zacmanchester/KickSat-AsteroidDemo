function [B, dB] = Magfit(theta)

ax0 = -7.397;
ax1 = -8.329;
bx1 =  46.7;
ax2 = -2.102;
bx2 = -10.77;

Bx = ax0 + ax1*cos(theta) + bx1*sin(theta) + ax2*cos(2*theta) + bx2*sin(2*theta);
dBx = -ax1*sin(theta) + bx1*cos(theta) - 2*ax2*sin(2*theta) + 2*bx2*cos(2*theta);

ay0 =  26.04;
ay1 = -23.6;
by1 = -10.09;
ay2 =  7.748;
by2 = -0.7238;

By = ay0 + ay1*cos(theta) + by1*sin(theta) + ay2*cos(2*theta) + by2*sin(2*theta);
dBy = -ay1*sin(theta) + by1*cos(theta) - 2*ay2*sin(2*theta) + 2*by2*cos(2*theta);

az0 =  8.147;
az1 = -9.776;
bz1 =  8.226;
az2 = -0.2044;
bz2 = -2.14;

Bz = az0 + az1*cos(theta) + bz1*sin(theta) + az2*cos(2*theta) + bz2*sin(2*theta);
dBz = -az1*sin(theta) + bz1*cos(theta) - 2*az2*sin(2*theta) + 2*bz2*cos(2*theta);
       
B = [Bx; By; Bz];
dB = [dBx; dBy; dBz];

end

