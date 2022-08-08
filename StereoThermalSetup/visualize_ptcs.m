%path = 'G:\Vista_project\finish1\no_rect_no_mtlQ_pyDisparsity_ptc\';
path = "G:\Vista_project\finish_ipc\combined-ptc\";
player3D = pcplayer([-20, 20], [-20, 20], [-100, 100], 'VerticalAxis', 'z', ...
    'VerticalAxisDir', 'up');
i=31;

while isOpen(player3D) 
     ptCloud = pcread(path+string(i)+".ply");
     if ptCloud.Count == 0
         continue
     end
     view(player3D,ptCloud);
     i = i + 1;
     
end 