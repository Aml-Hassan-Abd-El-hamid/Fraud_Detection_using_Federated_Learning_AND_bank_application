package com.Fintech.OnlineBanking.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import com.Fintech.OnlineBanking.domain.UserInformation;
import com.Fintech.OnlineBanking.projection.FullNameProjection;

public interface UserInformationRespository extends JpaRepository <UserInformation, Long> {
	UserInformation findByNationalId(Long nationalId);
	
}
