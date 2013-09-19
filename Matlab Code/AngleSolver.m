function theta = AngleSolver(B)

theta = pi;

for k = 1:5

    [Bg, dB] = Magfit(theta);
    e = B - Bg;
    e2 = e'*e
    de2dth = 2*e'*dB;
    
    alpha = 1;
    for j = 1:10
        thetanew = theta + alpha*e2/de2dth;
        Bnew = Magfit(thetanew);
        enew = B - Bnew;
        e2new = enew'*enew;
        if e2new < e2
            theta = thetanew
            break
        end
        alpha = .5*alpha;
    end 
end

end

